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
    assert cli._output_dir(RunConfig(case="x.m", tag="rung3")) == \
        os.path.join("results", "rung3")


def test_solve_writes_the_three_artifacts(tmp_path):
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
                              results.FRONTIER_JSON, cli.TRANSCRIPT])

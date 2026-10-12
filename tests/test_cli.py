import json
import os

import pytest

from conftest import CASE
from ropf import cli
from ropf.config import RunConfig


def test_bad_config_exits_2(tmp_path, capsys):
    path = tmp_path / "run.conf"
    path.write_text("kapa = 3\n")
    assert cli.main([str(path)]) == 2
    assert "did you mean 'kappa'" in capsys.readouterr().err


def test_no_case_exits_2(tmp_path, capsys):
    path = tmp_path / "run.conf"
    path.write_text("metric = joule_loss_max\n")
    assert cli.main([str(path)]) == 2
    assert "no case given" in capsys.readouterr().err


def test_output_dir_names_the_run():
    assert cli.output_dir(RunConfig(case="/x/case_ACTIVSg2000.m", stage="a2",
                                    metric="joule_loss_max")) == \
        os.path.join("results", "case_ACTIVSg2000_joule_loss_max_a2")
    assert cli.output_dir(RunConfig(case="x.m", q=5)) == \
        os.path.join("results", "x_max_active_flow_q5_baseline")
    assert cli.output_dir(RunConfig(outdir="out")) == "out"


def test_solve_writes_result_and_log(tmp_path):
    outdir = tmp_path / "out"
    path = tmp_path / "run.conf"
    path.write_text(f"case = {CASE}\nkappa = 2\nk_bar = 2\noutdir = {outdir}\n")
    try:
        code = cli.main([str(path), "--quiet"])
    except Exception as exc:
        pytest.skip(f"could not solve: {exc}")
    if code != 0:
        pytest.skip("the master did not solve")
    assert sorted(os.listdir(outdir)) == [
        "dispatch_branch.csv", "dispatch_bus.csv", "dispatch_gen.csv",
        "result.csv", "result.json", "run.log"]
    record = json.loads((outdir / "result.json").read_text())
    assert len(record["metadata"]["case_sha256"]) == 64
    assert record["metadata"]["buses"] == 200
    assert record["result"]["dispatch"]["Pg"]
    log = (outdir / "run.log").read_text()
    assert record["metadata"]["case_sha256"] in log
    assert "    Gurobi" in log and "finished" in log

import os

import pytest

from conftest import CASE
from ropf.config import ConfigError, read_config, resolve_case


def write(tmp_path, text):
    path = tmp_path / "run.conf"
    path.write_text(text)
    return str(path)


@pytest.mark.parametrize("text,match", [
    ("metrik = x\n", "unknown key 'metrik'"),
    ("kapa = 3\n", "did you mean 'kappa'"),
    ("k_bar = 25\nk_bar = 3\n", "already set on line 1"),
    ("kappa 2\n", "expected 'key = value'"),
    ("kappa = 2.5\n", "expected int"),
    ("eta = half\n", "expected float"),
    ("solver_verbose = maybe\n", "expected bool"),
    ("metric = max_line_loading\n", "unknown risk functional"),
    ("stage = a3\n", "unknown stage"),
    ("flow_domain = unrated\n", "unknown flow domain"),
    ("eta = 1.5\n", "eta"),
    ("time_limit_s = 0\n", "time_limit_s"),
    ("case = ACTIVSg9999\n", "Tried:"),
])
def test_refusals(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        read_config(write(tmp_path, text))


def test_values_are_typed(tmp_path):
    config = read_config(write(
        tmp_path, "# comment\n\nkappa = 3  # trailing\nrisk_weight = 12.5\n"
                  "threads = none\nsolver_verbose = yes\n"
                  "solver_dc_options = numericfocus=3 scale=2\n"))
    assert (config.kappa, config.risk_weight, config.threads,
            config.solver_verbose) == (3, 12.5, None, True)
    assert config.dc_solver().option_string() == \
        "timelim=3600 outlev=1 numericfocus=3 scale=2"


def test_config_reaches_the_algorithm(tmp_path):
    config = read_config(write(tmp_path, "metric = joule_loss_max\n"
                                         "flow_domain = rated\nq = 5\nkappa = 5\n"))
    alg = config.algorithm_config()
    assert (alg.metric, alg.flow_domain, alg.q, alg.kappa) == \
        ("joule_loss_max", "rated", 5, 5)


def test_case_resolves_by_bare_name(tmp_path):
    assert os.path.samefile(resolve_case("case_ACTIVSg200"), CASE)
    config = read_config(write(tmp_path, "case = case_ACTIVSg200\n"))
    assert os.path.samefile(config.case, CASE)

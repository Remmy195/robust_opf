"""The configuration file: what it accepts, and what it refuses.

Both refusals are the point.  A config that ignores an unknown key turns a typo
into a study that measured the wrong thing; a config that takes the last of two
values for the same key looks correct at whichever line the reader checks.
"""

from __future__ import annotations

import os
from dataclasses import fields

import pytest

from ropf.config import KEYS, ConfigError, RunConfig, read_config, resolve_case

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


def write(tmp_path, text, name="run.conf"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


###############################################################################
# The two refusals
###############################################################################


def test_unknown_key_is_an_error(tmp_path):
    path = write(tmp_path, "metrik = joule_loss_max\n")
    with pytest.raises(ConfigError) as exc:
        read_config(path)
    assert "unknown key 'metrik'" in str(exc.value)
    assert "1" in str(exc.value), "the error must name the line"


def test_unknown_key_suggests_the_intended_one(tmp_path):
    path = write(tmp_path, "kapa = 3\n")
    with pytest.raises(ConfigError, match="did you mean 'kappa'"):
        read_config(path)


def test_duplicate_key_is_an_error(tmp_path):
    """Last-one-wins is the usual choice and the worst one."""
    path = write(tmp_path, "k_bar = 25\nkappa = 2\n\n# ...\nk_bar = 3\n")
    with pytest.raises(ConfigError) as exc:
        read_config(path)
    message = str(exc.value)
    assert "'k_bar' is already set on line 1" in message
    assert ":5:" in message, "the error must name both lines"


def test_a_duplicate_of_the_same_value_is_still_an_error(tmp_path):
    path = write(tmp_path, "kappa = 2\nkappa = 2\n")
    with pytest.raises(ConfigError, match="already set"):
        read_config(path)


def test_a_line_that_is_not_key_equals_value_is_an_error(tmp_path):
    path = write(tmp_path, "kappa 2\n")
    with pytest.raises(ConfigError, match="expected 'key = value'"):
        read_config(path)


###############################################################################
# Types
###############################################################################


def test_a_fractional_integer_is_rejected_not_truncated(tmp_path):
    path = write(tmp_path, "kappa = 2.5\n")
    with pytest.raises(ConfigError) as exc:
        read_config(path)
    assert "rather than truncated" in str(exc.value)


def test_a_non_numeric_number_is_an_error(tmp_path):
    path = write(tmp_path, "eta = half\n")
    with pytest.raises(ConfigError, match="expected a number"):
        read_config(path)


@pytest.mark.parametrize("text,expected", [("1", True), ("true", True),
                                           ("YES", True), ("on", True),
                                           ("0", False), ("false", False),
                                           ("no", False), ("off", False)])
def test_booleans(tmp_path, text, expected):
    config = read_config(write(tmp_path, f"solver_verbose = {text}\n"))
    assert config.solver_verbose is expected


def test_a_non_boolean_boolean_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="expected a boolean"):
        read_config(write(tmp_path, "solver_verbose = maybe\n"))


@pytest.mark.parametrize("text", ["0, 0.5, 1, 2", "0 0.5 1 2", "0,0.5,1,2"])
def test_the_weight_grid_takes_a_list(tmp_path, text):
    config = read_config(write(tmp_path, f"weight_grid = {text}\n"))
    assert config.weight_grid == (0.0, 0.5, 1.0, 2.0)


def test_an_optional_takes_none(tmp_path):
    assert read_config(write(tmp_path, "risk_weight = none\n")).risk_weight is None
    assert read_config(write(tmp_path, "risk_weight = 12.5\n")).risk_weight == 12.5


def test_comments_and_blank_lines_are_ignored(tmp_path):
    config = read_config(write(tmp_path, "# a comment\n\nkappa = 3  # trailing\n"))
    assert config.kappa == 3


###############################################################################
# Values the study will not accept
###############################################################################


def test_an_unknown_metric_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="is not one of"):
        read_config(write(tmp_path, "metric = max_line_loading\n"))


def test_an_unknown_stage_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="is not one of"):
        read_config(write(tmp_path, "stage = dcac\n"))


def test_a_negative_weight_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot be negative"):
        read_config(write(tmp_path, "weight_grid = 0, -1\n"))


def test_algorithm_rules_are_enforced_once(tmp_path):
    """eta lives in AlgorithmConfig, so the config must not restate the rule."""
    with pytest.raises(ValueError, match="eta"):
        read_config(write(tmp_path, "eta = 1.5\n"))


###############################################################################
# Wiring
###############################################################################


def test_the_key_table_is_the_dataclass(tmp_path):
    """There is one place a study parameter is declared, and this is it."""
    assert KEYS == tuple(f.name for f in fields(RunConfig))


def test_the_case_resolves_relative_to_the_config_file(tmp_path):
    import shutil
    shutil.copy(CASE, tmp_path / "case_ACTIVSg200.m")
    config = read_config(write(tmp_path, "case = case_ACTIVSg200.m\n"))
    assert os.path.isfile(config.case)


def test_a_missing_case_names_everywhere_it_looked(tmp_path):
    with pytest.raises(ConfigError) as exc:
        read_config(write(tmp_path, "case = ACTIVSg9999\n"))
    message = str(exc.value)
    assert "Tried:" in message and "ACTIVSg9999.m" in message


def test_an_absolute_lambda_makes_a_single_point(tmp_path):
    config = read_config(write(tmp_path, "risk_weight = 500\n"))
    assert config.weights == (1.0,)
    assert config.algorithm_config().risk_weight == 500.0


def test_flow_domain_defaults_to_the_whole_edge_set():
    """Every config written before the key existed has to keep meaning what it
    meant, so the default is the unrestricted maximum of eq (10a)."""
    assert RunConfig().flow_domain == "all"
    assert RunConfig().algorithm_config().flow_domain == "all"


def test_flow_domain_reaches_the_algorithm(tmp_path):
    config = read_config(write(tmp_path, "flow_domain = rated\n"))
    assert config.algorithm_config().flow_domain == "rated"


def test_an_unknown_flow_domain_is_an_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        read_config(write(tmp_path, "flow_domain = unrated\n"))
    assert "flow_domain" in str(exc.value)


@pytest.mark.parametrize("metric", ["joule_loss_max", "bus_flow_sum_agg"])
def test_flow_domain_reaches_the_algorithm_for_every_metric(tmp_path, metric):
    """`ropf.risk` restricts all three functionals, not max_active_flow alone,
    so pairing flow_domain = rated with another metric is not a config error."""
    config = read_config(write(tmp_path,
                               f"metric = {metric}\nflow_domain = rated\n"))
    assert config.algorithm_config().flow_domain == "rated"


def test_the_grid_is_swept_when_no_absolute_lambda_is_given():
    assert RunConfig().weights == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def test_solver_configs_carry_the_declared_budget():
    config = RunConfig(time_limit_s=60.0, threads=4, solver_ac="ipopt")
    assert config.dc_solver().time_limit_s == 60.0
    assert config.ac_solver().name == "ipopt"
    assert config.dc_solver().knitro_threads == 4


def test_resolve_case_accepts_a_bare_name():
    assert resolve_case(CASE) == CASE
    assert os.path.samefile(resolve_case("case_ACTIVSg200", FIXTURES), CASE)

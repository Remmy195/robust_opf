"""eq (weightstar): the weight as an exchange rate on lambda*.

    lambda = xi/(tau_hi - tau_lo) * lambda*,   lambda* = z0/rho0

The two halves are tested apart, because they fail for different reasons.
lambda* carries the SCALE of the system and is only knowable after the nominal
solve; the exchange rate is a dimensionless STUDY PARAMETER, knowable before
anything runs, and is what the weight grid is a grid of.
"""

from __future__ import annotations

import os

import pytest

from ropf import algorithm
from ropf.algorithm import WEIGHT_GRID, AlgorithmConfig, exchange_rate
from ropf.config import ConfigError, RunConfig, read_config

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


def write(tmp_path, text, name="run.conf"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


###############################################################################
# The rate itself
###############################################################################


@pytest.mark.parametrize("xi, tau_lo, tau_hi, rate", [
    (0.05, 0.5, 0.7, 0.25),      # the grid's bottom non-zero point
    (0.10, 0.5, 0.7, 0.5),
    (0.20, 0.5, 0.7, 1.0),
    (0.40, 0.5, 0.7, 2.0),
    (0.80, 0.5, 0.7, 4.0),       # the grid's top point
    (0.10, 0.4, 0.9, 0.2),
    (1.00, 0.0 + 1e-12, 1.0, 1.0),
])
def test_the_rate_is_xi_over_the_band_width(xi, tau_lo, tau_hi, rate):
    assert exchange_rate(xi, tau_lo, tau_hi) == pytest.approx(rate)


@pytest.mark.parametrize("xi, tau_lo, tau_hi, exact", [
    (0.05, 0.5, 0.7, 0.25),
    (0.10, 0.5, 0.7, 0.5),
    (0.20, 0.5, 0.7, 1.0),
    (0.40, 0.5, 0.7, 2.0),
    (0.80, 0.5, 0.7, 4.0),
])
def test_the_rate_is_exactly_the_multiplier_not_one_ulp_off(xi, tau_lo, tau_hi,
                                                            exact):
    """The same study written two ways must produce the same number, exactly.

    In binary 0.05/(0.7 - 0.5) is 0.25000000000000006, and that sixteenth-digit
    difference is not harmless: MEASURED on ACTIVSg200 under joule_loss_max it
    moved the reported risk by 2.4e-6 relative, because where the master has
    many optima the barrier picks among them on numerical detail alone.  An
    equality test, not approx, because approx is exactly what would let this
    regress.
    """
    assert exchange_rate(xi, tau_lo, tau_hi) == exact


def test_the_tolerance_grid_reproduces_the_weight_grid_exactly():
    """eq (weightgrid) stated as tolerances IS eq (weightgrid)."""
    band = (0.5, 0.7)
    rates = tuple(exchange_rate(x, *band)
                  for x in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8))
    assert rates == WEIGHT_GRID


def test_rounding_does_not_disturb_a_rate_that_is_already_clean():
    """Twelve digits is far beyond any weight anyone states, so a rate given
    to a few digits must come back untouched."""
    for xi, lo, hi, expected in [(0.3, 0.4, 0.7, 1.0),
                                 (0.123, 0.5, 0.6, 1.23),
                                 (0.075, 0.2, 0.8, 0.125)]:
        assert exchange_rate(xi, lo, hi) == expected


def test_a_zero_tolerance_prices_risk_at_zero():
    """xi = 0 pays nothing for risk, which is the lambda = 0 point of the grid."""
    assert exchange_rate(0.0, 0.5, 0.9) == 0.0


def test_a_wider_band_prices_risk_lower():
    """The band is the denominator: more risk removed for the same tolerance
    is a lower price per unit, not a higher one."""
    narrow = exchange_rate(0.1, 0.5, 0.6)
    wide = exchange_rate(0.1, 0.5, 0.9)
    assert wide < narrow
    assert narrow == pytest.approx(4.0 * wide)


def test_the_rate_is_dimensionless_in_both_arguments():
    """Scaling xi scales the rate; the band enters only through its width."""
    assert exchange_rate(0.2, 0.5, 0.7) == \
        pytest.approx(2.0 * exchange_rate(0.1, 0.5, 0.7))
    assert exchange_rate(0.1, 0.5, 0.7) == \
        pytest.approx(exchange_rate(0.1, 0.2, 0.4)), \
        "only the width of the band may matter, not where it sits"


###############################################################################
# What the rate refuses
###############################################################################


def test_a_negative_tolerance_is_an_error():
    with pytest.raises(ValueError, match="xi"):
        exchange_rate(-0.1, 0.5, 0.9)


@pytest.mark.parametrize("tau_lo, tau_hi", [
    (0.7, 0.5),      # inverted: the width is negative, so is the price
    (0.5, 0.5),      # empty: the width is zero, and the price divides by it
    (0.0, 0.9),      # tau_lo = 0 is excluded by 0 < tau_lo
    (-0.1, 0.9),
    (0.5, 1.1),      # more risk than the nominal dispatch carries
    (1.0, 1.0),
])
def test_a_band_outside_the_unit_interval_is_an_error(tau_lo, tau_hi):
    with pytest.raises(ValueError, match="band"):
        exchange_rate(0.1, tau_lo, tau_hi)


def test_an_empty_band_never_divides_by_zero():
    """The refusal comes before the division, not as a ZeroDivisionError."""
    with pytest.raises(ValueError, match="band"):
        exchange_rate(0.1, 0.5, 0.5)


###############################################################################
# The grid IS a grid of exchange rates
###############################################################################


def test_every_grid_point_is_a_reachable_exchange_rate():
    """eq (weightgrid) is stated in the manuscript as exchange rates, so each
    of its multipliers must be expressible as one."""
    band = (0.5, 0.7)
    for multiplier in WEIGHT_GRID:
        xi = multiplier * (band[1] - band[0])
        assert exchange_rate(xi, *band) == pytest.approx(multiplier)


def test_the_grid_spans_a_quarter_to_four_units_of_cost_per_unit_of_risk():
    """The reading the manuscript's OPEN note gives the grid."""
    nonzero = [m for m in WEIGHT_GRID if m > 0]
    assert min(nonzero) == pytest.approx(0.25)
    assert max(nonzero) == pytest.approx(4.0)


###############################################################################
# The config keys
###############################################################################


def test_the_triple_sets_the_weight(tmp_path):
    config = read_config(write(tmp_path,
                               "xi = 0.2\ntau_lo = 0.5\ntau_hi = 0.7\n"))
    assert config.exchange_rate == pytest.approx(1.0)
    assert config.weights == pytest.approx((1.0,)), \
        "one (xi, band) triple states one price, so it is a single point"


def test_the_triple_is_the_multiplier_the_algorithm_is_given(tmp_path):
    config = read_config(write(tmp_path,
                               "xi = 0.1\ntau_lo = 0.5\ntau_hi = 0.7\n"))
    assert config.exchange_rate == pytest.approx(0.5)
    algorithm_config = config.algorithm_config(config.weights[0])
    assert algorithm_config.weight_multiplier == pytest.approx(0.5)
    assert algorithm_config.risk_weight is None, \
        "the rate is a multiple of lambda*, not an absolute lambda"


def test_xi_grid_sweeps_one_rate_per_tolerance(tmp_path):
    """A band plus a grid of tolerances is a frontier stated in the operator's
    own terms: how much extra cost am I willing to pay, at each point."""
    config = read_config(write(tmp_path, "xi_grid = 0, 0.05, 0.1, 0.2, 0.4, 0.8\n"
                                         "tau_lo = 0.5\ntau_hi = 0.7\n"))
    assert config.weights == pytest.approx(WEIGHT_GRID), \
        "at a band of width 0.2 this IS eq (weightgrid)"


def test_xi_grid_and_a_multiplier_grid_are_the_same_study(tmp_path):
    """The rate is the multiplier, so the two ways of saying it must agree."""
    by_xi = read_config(write(tmp_path, "xi_grid = 0, 0.05, 0.1, 0.2, 0.4, 0.8\n"
                                        "tau_lo = 0.5\ntau_hi = 0.7\n"))
    by_multiplier = read_config(write(tmp_path,
                                      "weight_grid = 0, 0.25, 0.5, 1, 2, 4\n",
                                      name="b.conf"))
    assert by_xi.weights == pytest.approx(by_multiplier.weights)


def test_a_wider_band_moves_the_whole_curve_down(tmp_path):
    """Same tolerances, wider band: every weight is lower, so every point of
    the frontier is cheaper and less de-risked."""
    narrow = RunConfig(xi_grid=(0.1, 0.2), tau_lo=0.5, tau_hi=0.6)
    wide = RunConfig(xi_grid=(0.1, 0.2), tau_lo=0.5, tau_hi=0.9)
    assert all(w < n for w, n in zip(wide.weights, narrow.weights))


def test_xi_grid_without_a_band_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="band"):
        read_config(write(tmp_path, "xi_grid = 0, 0.1, 0.2\n"))


def test_xi_and_xi_grid_may_not_both_be_set(tmp_path):
    with pytest.raises(ConfigError, match="no defensible answer"):
        read_config(write(tmp_path, "xi = 0.2\nxi_grid = 0, 0.1\n"
                                    "tau_lo = 0.5\ntau_hi = 0.7\n"))


def test_a_negative_tolerance_in_the_grid_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="xi"):
        read_config(write(tmp_path, "xi_grid = 0, -0.1\n"
                                    "tau_lo = 0.5\ntau_hi = 0.7\n"))


def test_no_triple_leaves_the_grid_alone():
    config = RunConfig()
    assert config.exchange_rate is None
    assert config.exchange_rates == ()
    assert config.weights == WEIGHT_GRID


@pytest.mark.parametrize("text, names_the_gap", [
    ("xi = 0.2\n", "missing"),
    ("xi_grid = 0, 0.2\n", "missing"),
    ("xi = 0.2\ntau_lo = 0.5\n", "incomplete"),
    ("xi = 0.2\ntau_hi = 0.7\n", "incomplete"),
    ("tau_lo = 0.5\n", "neither xi nor xi_grid"),
    ("tau_hi = 0.7\n", "neither xi nor xi_grid"),
    ("tau_lo = 0.5\ntau_hi = 0.7\n", "neither xi nor xi_grid"),
])
def test_a_partial_statement_is_an_error_not_a_fallback(tmp_path, text,
                                                        names_the_gap):
    """xi has no denominator on its own and the band has no numerator; the
    only way to carry on would be to invent the missing half.  The message
    must name WHICH half is missing, since that is the whole fix."""
    with pytest.raises(ConfigError, match=names_the_gap):
        read_config(write(tmp_path, text))


def test_the_triple_and_an_absolute_lambda_may_not_both_be_set(tmp_path):
    """Both set the weight, and which applies has no defensible answer."""
    with pytest.raises(ConfigError, match="no defensible answer"):
        read_config(write(tmp_path, "risk_weight = 500\nxi = 0.2\n"
                                    "tau_lo = 0.5\ntau_hi = 0.7\n"))


def test_a_bad_band_is_refused_when_the_file_is_read(tmp_path):
    with pytest.raises(ConfigError, match="band"):
        read_config(write(tmp_path, "xi = 0.2\ntau_lo = 0.7\ntau_hi = 0.5\n"))


def test_the_triple_is_recorded_in_the_run_record(tmp_path):
    """A run's record must say what price it priced at, and what built it."""
    config = read_config(write(tmp_path,
                               "xi = 0.2\ntau_lo = 0.5\ntau_hi = 0.7\n"))
    record = config.as_dict()
    assert (record["xi"], record["tau_lo"], record["tau_hi"]) == (0.2, 0.5, 0.7)


def test_the_keys_are_declared_in_the_key_table():
    from ropf.config import KEYS
    for key in ("xi", "tau_lo", "tau_hi"):
        assert key in KEYS


def test_the_tracer_refuses_a_pinned_rate(tmp_path):
    """A pinned rate IS a pinned multiplier, so there is nothing to sweep."""
    from ropf.study import frontier

    config = read_config(write(tmp_path, f"case = {CASE}\nxi = 0.2\n"
                                         f"tau_lo = 0.5\ntau_hi = 0.7\n"))
    with pytest.raises(ConfigError, match="pins the exchange rate"):
        frontier.trace(config, None, object())


###############################################################################
# lambda* is the other half, and is unchanged
###############################################################################


def test_lambda_star_is_still_z0_over_rho0():
    """The rate multiplies lambda*; it does not replace it.

    eq (weightstar) has two halves and only the factor in front is new, so the
    reference weight the loop computes must be exactly what it always was.
    """
    source = __import__("inspect").getsource(algorithm.run_loop)
    assert "result.z0 / rho0" in source


@pytest.mark.parametrize("xi, tau_lo, tau_hi, expected_lambda", [
    (0.05, 0.5, 0.7, 0.25 * 7391.7),
    (0.20, 0.5, 0.7, 1.00 * 7391.7),
    (0.80, 0.5, 0.7, 4.00 * 7391.7),
])
def test_lambda_is_the_rate_times_lambda_star(xi, tau_lo, tau_hi,
                                              expected_lambda):
    """The whole of eq (weightstar), assembled from its two halves.

    z0 and rho0 are the real nominal quantities of ACTIVSg200 under
    max_active_flow, so lambda* = z0/rho0 is the real scale of that system and
    these are the dollars per p.u. the loop would actually price risk at.
    """
    z0, rho0 = 27479.643306, 3.71790
    lambda_star = z0 / rho0
    assert lambda_star == pytest.approx(7391.7, rel=1e-4)

    rate = exchange_rate(xi, tau_lo, tau_hi)
    assert rate * lambda_star == pytest.approx(expected_lambda, rel=1e-4)
    # And that is the number the algorithm carries: the rate goes in as the
    # multiplier, and the loop multiplies it by the lambda* it computes.
    assert AlgorithmConfig(weight_multiplier=rate).weight_multiplier == \
        pytest.approx(rate)

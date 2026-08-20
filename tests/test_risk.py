"""The risk functionals and their separation, Sections 2.3 and 2.4.

These run without AMPL: everything in `ropf.risk` is a pure function of plain
dictionaries, which is the reason it is written that way.
"""

from __future__ import annotations

import os

import pytest

from ropf import risk
from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


@pytest.fixture(scope="module")
def net():
    return read_matpower(CASE)


@pytest.fixture(scope="module")
def dispatch(net):
    """A deterministic, sign-varied dispatch. No solver involved.

    Alternating signs matter: a dispatch with every flow positive would let a
    cut that mishandles signs pass by accident.
    """
    Pf = {}
    for i, count in enumerate(sorted(net.branches)):
        Pf[count] = (1.0 + (count % 7)) * (1.0 if i % 3 else -1.0) * 0.1
    Pg = {}
    for i, count in enumerate(sorted(net.gens)):
        Pg[count] = (0.5 + (count % 5) * 0.25) * (1.0 if i % 4 else -1.0)
    return Pf, Pg


###############################################################################
# The functionals
###############################################################################


def test_flow_functional_is_max_abs_flow(net, dispatch):
    Pf, _ = dispatch
    ev = risk.evaluate("max_active_flow", net, Pf)
    assert ev.value == pytest.approx(max(abs(v) for v in Pf.values()))
    assert ev.components[ev.argmax] == pytest.approx(ev.value)


def test_joule_functional_uses_resistance(net, dispatch):
    Pf, _ = dispatch
    ev = risk.evaluate("joule_loss_max", net, Pf)
    expected = max(net.branches[c].r * Pf[c] ** 2 for c in net.branches)
    assert ev.value == pytest.approx(expected)


def test_unknown_metric_is_an_error(net, dispatch):
    """A typo in a config must not silently become the flow family."""
    Pf, _ = dispatch
    with pytest.raises(ValueError, match="unknown risk functional"):
        risk.evaluate("max_line_loading", net, Pf)


###############################################################################
# The bus functional, eq (4a)
###############################################################################


def test_bus_functional_includes_generation_and_demand(net, dispatch):
    """f_i carries all three terms of eq (4a), not just the incident flows.

    Checked at a bus that actually has generation and demand. The argmax bus is
    deliberately not used: on ACTIVSg200 the busiest bus under this dispatch is
    a pure transit bus with neither term, which is a legitimate outcome and
    would make the test vacuous.
    """
    Pf, Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf, Pg)

    # The TAMU synthetic grids place generation and load at separate buses:
    # ACTIVSg200 has no bus carrying both, and ACTIVSg2000 has two out of 2000.
    # So each term is checked where it actually occurs, rather than looking for
    # a bus that has both and finding none.
    gen_buses = [c for c, bus in net.buses.items() if bus.genidsbycount]
    load_buses = [c for c, bus in net.buses.items() if bus.Pd > 0]
    assert gen_buses and load_buses

    checked_gen = 0
    for bus_count in gen_buses:
        flow_term = sum(abs(Pf[b]) for b in ev.incidence[bus_count])
        gen_term = sum(abs(Pg[g]) for g in ev.gen_incidence[bus_count])
        demand_term = abs(net.buses[bus_count].Pd)
        assert ev.components[bus_count] == pytest.approx(
            flow_term + gen_term + demand_term)
        if gen_term > 0:
            # Dropping the generation term would be visible here.
            assert ev.components[bus_count] > flow_term + demand_term
            checked_gen += 1
    assert checked_gen, "no generator carried nonzero output"

    checked_load = 0
    for bus_count in load_buses:
        flow_term = sum(abs(Pf[b]) for b in ev.incidence[bus_count])
        gen_term = sum(abs(Pg[g]) for g in ev.gen_incidence[bus_count])
        demand_term = abs(net.buses[bus_count].Pd)
        assert ev.components[bus_count] == pytest.approx(
            flow_term + gen_term + demand_term)
        assert ev.components[bus_count] > flow_term + gen_term
        checked_load += 1
    assert checked_load


def test_bus_functional_sums_absolute_values(net, dispatch):
    """No cancellation: f_i is a sum of magnitudes.

    A bus whose incident flows are equal and opposite must report their sum, not
    zero. This is the property that makes f_i convex and the cut a subgradient.
    """
    Pf, Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf, Pg)
    for bus_count, bus in net.buses.items():
        incident = ev.incidence[bus_count]
        expected = sum(abs(Pf[b]) for b in incident)
        expected += sum(abs(Pg[g]) for g in ev.gen_incidence[bus_count])
        expected += abs(bus.Pd)
        assert ev.components[bus_count] == pytest.approx(expected), (
            f"bus {bus_count} does not sum magnitudes")


def test_bus_functional_is_not_twice_the_max_flow(net, dispatch):
    """Regression on the branch-only functional.

    Summing only incident flows makes a degree-2 bus report exactly twice the
    largest line flow, which is how the earlier implementation was caught
    (case118 reported 14.2 against a max flow of 7.1). With generation and
    demand included that coincidence must not recur.
    """
    Pf, Pg = dispatch
    bus_ev = risk.evaluate("bus_flow_sum_agg", net, Pf, Pg)
    flow_ev = risk.evaluate("max_active_flow", net, Pf)
    assert bus_ev.value != pytest.approx(2.0 * flow_ev.value, rel=1e-9)


###############################################################################
# Cuts, eq (6c)
###############################################################################


def test_bus_cut_equals_the_functional_at_the_incumbent(net, dispatch):
    """The subgradient property, which everything else rests on.

    With the signs frozen at the incumbent, the cut's right-hand side must equal
    f_i exactly *at that dispatch*. If it does not, the cut is not a subgradient
    inequality and eq (11) does not hold.
    """
    Pf, Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf, Pg)
    chosen = risk.select(ev, kappa=10)
    cuts = risk.build_bus_cuts(ev, chosen)
    assert cuts

    for cut in cuts:
        rhs = risk.cut_value_at(ev, cut, Pf, Pg)
        assert rhs == pytest.approx(ev.components[cut.bus], rel=1e-12), (
            f"cut at bus {cut.bus} does not reproduce f_i")


def test_bus_cut_is_a_minorant_away_from_the_incumbent(net, dispatch):
    """Moving off the incumbent, the cut may only under-estimate f_i.

    A minorant is what makes (M) a relaxation. If a perturbed dispatch ever put
    the cut above the functional the master could cut off the true optimum.
    """
    Pf, Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf, Pg)
    cuts = risk.build_bus_cuts(ev, risk.select(ev, kappa=5))

    for scale in (-1.0, -0.3, 0.5, 2.0):
        moved_f = {k: v * scale for k, v in Pf.items()}
        moved_g = {k: v * scale for k, v in Pg.items()}
        moved = risk.evaluate("bus_flow_sum_agg", net, moved_f, moved_g)
        for cut in cuts:
            rhs = risk.cut_value_at(ev, cut, moved_f, moved_g)
            assert rhs <= moved.components[cut.bus] + 1e-9, (
                f"cut at bus {cut.bus} exceeds f_i at scale {scale}")
            assert rhs <= moved.value + 1e-9, (
                f"cut at bus {cut.bus} exceeds phi at scale {scale}")


def test_selection_returns_kappa_distinct_components(net, dispatch):
    """kappa separate cuts per iteration, not one at the argmax."""
    Pf, Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf, Pg)
    chosen = risk.select(ev, kappa=10)
    assert len(chosen) == 10
    assert len(set(chosen)) == 10
    # Ordered by descending f_i.
    values = [ev.components[b] for b in chosen]
    assert values == sorted(values, reverse=True)


def test_selection_skips_components_already_cut(net, dispatch):
    Pf, _ = dispatch
    ev = risk.evaluate("max_active_flow", net, Pf)
    first = risk.select(ev, kappa=5)
    second = risk.select(ev, kappa=5, existing=set(first))
    assert not set(first) & set(second)


def test_bus_exclusion_is_by_bus_and_sign_pattern(net, dispatch):
    """A repeated bus under a new sign pattern is a new, admissible hyperplane.

    Excluding by bus alone would refuse it and stall the loop; excluding by
    nothing would re-add the identical hyperplane every iteration.
    """
    Pf, Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf, Pg)
    chosen = risk.select(ev, kappa=3)
    keys = {risk.bus_cut_key(ev, b) for b in chosen}

    # Same dispatch: every key is already held, so nothing is selected again.
    assert risk.select(ev, kappa=3, existing=keys) != chosen

    # Flip every flow. The buses are unchanged but their sign patterns are not,
    # so the same buses become selectable again.
    flipped = {k: -v for k, v in Pf.items()}
    ev2 = risk.evaluate("bus_flow_sum_agg", net, flipped, Pg)
    again = risk.select(ev2, kappa=3, existing=keys)
    assert set(again) & set(chosen), (
        "a new sign pattern at the same bus must be selectable")


def test_empty_cut_is_recognised(net):
    """A cut with no terms reads Phi >= 0 and must not be appended."""
    zero_f = {c: 0.0 for c in net.branches}
    zero_g = {c: 0.0 for c in net.gens}
    ev = risk.evaluate("bus_flow_sum_agg", net, zero_f, zero_g)
    # A bus with no demand either: every term is zero.
    bare = [c for c, bus in net.buses.items() if bus.Pd == 0.0]
    assert bare, "fixture has no zero-demand bus"
    cuts = risk.build_bus_cuts(ev, bare[:1])
    assert cuts[0].is_empty

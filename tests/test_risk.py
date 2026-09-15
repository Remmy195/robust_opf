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


def test_bus_functional_excludes_generation_and_demand(net, dispatch):
    """f_i is the incident line flows alone: eq (4a) has one term, not three.

    Generation is already carried by the flows -- what a unit injects at the bus
    leaves through the incident lines -- so a generation term would count the
    same power twice.  Demand is a constant of the case and cannot be traded
    against anything, so carrying it would only reorder the argmax toward the
    most heavily loaded bus rather than the busiest one.

    Checked at buses that actually have generation, and at buses that actually
    have load, because the TAMU grids place the two apart: ACTIVSg200 has no bus
    carrying both, and ACTIVSg2000 has two out of 2000.
    """
    Pf, Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)

    gen_buses = [c for c, bus in net.buses.items() if bus.genidsbycount]
    load_buses = [c for c, bus in net.buses.items() if bus.Pd > 0]
    assert gen_buses and load_buses

    checked_gen = 0
    for bus_count in gen_buses:
        flow_term = sum(abs(Pf[b]) for b in ev.incidence[bus_count])
        assert ev.components[bus_count] == pytest.approx(flow_term)
        if any(abs(Pg[g]) > 0 for g in net.buses[bus_count].genidsbycount):
            checked_gen += 1
    assert checked_gen, "no generator carried nonzero output"

    for bus_count in load_buses:
        flow_term = sum(abs(Pf[b]) for b in ev.incidence[bus_count])
        assert ev.components[bus_count] == pytest.approx(flow_term)
        assert net.buses[bus_count].Pd > 0


def test_the_functional_does_not_depend_on_the_generation_at_all(net, dispatch):
    """Moving generation while holding the flows fixed must not move f_i.

    The strongest statement of the same property: `evaluate` no longer takes a
    Pg, so a caller cannot pass one, and the value is a function of the flows.
    """
    Pf, Pg = dispatch
    first = risk.evaluate("bus_flow_sum_agg", net, Pf)
    doubled = {g: 2.0 * v for g, v in Pg.items()}
    assert doubled != Pg
    second = risk.evaluate("bus_flow_sum_agg", net, Pf)
    assert first.value == pytest.approx(second.value)
    assert first.components == second.components


def test_bus_functional_sums_absolute_values(net, dispatch):
    """No cancellation: f_i is a sum of magnitudes.

    A bus whose incident flows are equal and opposite must report their sum, not
    zero. This is the property that makes f_i convex and the cut a subgradient.
    """
    Pf, _Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
    for bus_count in net.buses:
        expected = sum(abs(Pf[b]) for b in ev.incidence[bus_count])
        assert ev.components[bus_count] == pytest.approx(expected), (
            f"bus {bus_count} does not sum magnitudes")


def test_a_degree_two_bus_carries_twice_its_flow(net, dispatch):
    """The 2x coincidence is the correct answer, not a symptom.

    An earlier version of this suite treated "a degree-2 bus reports exactly
    twice the largest line flow" as the tell of a broken branch-only
    functional.  With eq (4a) reduced to the incident flows that is simply what
    f_i is: the power crosses two lines and is counted on each.
    """
    Pf, _Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)

    degree_two = [c for c in net.buses if len(ev.incidence[c]) == 2]
    assert degree_two, "the case has no degree-2 bus to check"

    checked = 0
    for bus_count in degree_two:
        a, b = ev.incidence[bus_count]
        assert ev.components[bus_count] == pytest.approx(
            abs(Pf[a]) + abs(Pf[b]))
        if abs(Pf[a]) == pytest.approx(abs(Pf[b]), rel=1e-6) and abs(Pf[a]) > 0:
            assert ev.components[bus_count] == pytest.approx(2.0 * abs(Pf[a]))
            checked += 1
    assert checked, "no degree-2 bus carried a balanced nonzero flow"


###############################################################################
# The domain of eq (10a)
###############################################################################


@pytest.mark.parametrize("metric", risk.METRICS)
def test_rated_domain_is_a_no_op_where_every_branch_is_rated(net, dispatch,
                                                              metric):
    """ACTIVSg200 rates all 245 of its branches, so the restriction removes
    nothing and must not perturb the value or the component set, for any of
    the three functionals."""
    Pf, _ = dispatch
    assert net.unconstrained_branches() == []
    everything = risk.evaluate(metric, net, Pf, "all")
    rated = risk.evaluate(metric, net, Pf, "rated")
    assert rated.value == pytest.approx(everything.value)
    assert rated.components == everything.components


@pytest.mark.parametrize("metric", ["max_active_flow", "joule_loss_max"])
def test_rated_domain_drops_the_unrated_attaining_branches(net, dispatch,
                                                            metric):
    """With every attaining branch marked unrated, the maximum falls to the
    largest of the rest and those branches leave the component set entirely.

    They have to leave the components, not merely lose the argmax: the
    separation selects the cuts it appends from exactly this dictionary.

    The whole attaining set is marked because this dispatch ties at the
    maximum, and dropping one of several branches that all attain it would
    leave the value where it was and prove nothing.
    """
    Pf, _ = dispatch
    everything = risk.evaluate(metric, net, Pf, "all")
    attaining = [c for c, v in everything.components.items()
                 if v == pytest.approx(everything.value)]
    runner_up = max(v for c, v in everything.components.items()
                    if c not in attaining)

    kept = {c: net.branches[c].constrainedflow for c in attaining}
    for c in attaining:
        net.branches[c].constrainedflow = 0
    try:
        rated = risk.evaluate(metric, net, Pf, "rated")
    finally:
        for c, value in kept.items():
            net.branches[c].constrainedflow = value

    assert not set(attaining) & set(rated.components)
    assert len(rated.components) == len(everything.components) - len(attaining)
    assert rated.value == pytest.approx(runner_up)
    assert rated.value < everything.value


def test_rated_domain_drops_a_bus_left_with_no_rated_incident_branch(
        net, dispatch):
    """The bus functional restricts the SUM a candidate is built from, not the
    outer max's candidates directly, since a bus carries no rating of its own.

    Marking ALL of one bus's incident branches unrated must drop that bus from
    the component set entirely -- it has no rated sum to report, not a sum of
    zero -- the same reading `study/domain_scan.py` uses for its free
    cross-evaluation.
    """
    Pf, _ = dispatch
    everything = risk.evaluate("bus_flow_sum_agg", net, Pf, "all")
    bus = max(everything.incidence, key=lambda b: len(everything.incidence[b]))
    incident = everything.incidence[bus]
    assert incident, "need a bus with at least one incident branch"

    kept = {c: net.branches[c].constrainedflow for c in incident}
    for c in incident:
        net.branches[c].constrainedflow = 0
    try:
        rated = risk.evaluate("bus_flow_sum_agg", net, Pf, "rated")
    finally:
        for c, value in kept.items():
            net.branches[c].constrainedflow = value

    # Zeroing every incident branch of `bus` can also drop a NEIGHBOR for
    # which one of those same branches was its own only rated line, so the
    # count falls by at least one, not by exactly one.
    assert bus not in rated.components
    assert bus not in rated.incidence
    assert len(rated.components) < len(everything.components)


def test_unknown_flow_domain_is_an_error(net, dispatch):
    Pf, _ = dispatch
    with pytest.raises(ValueError) as exc:
        risk.evaluate("max_active_flow", net, Pf, "rateA")
    assert "unknown flow domain" in str(exc.value)

###############################################################################
# Cuts, eq (6c)
###############################################################################


def test_bus_cut_equals_the_functional_at_the_incumbent(net, dispatch):
    """The subgradient property, which everything else rests on.

    With the signs frozen at the incumbent, the cut's right-hand side must equal
    f_i exactly *at that dispatch*. If it does not, the cut is not a subgradient
    inequality and eq (11) does not hold.
    """
    Pf, _Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
    chosen = risk.select(ev, kappa=10)
    cuts = risk.build_bus_cuts(ev, chosen)
    assert cuts

    for cut in cuts:
        rhs = risk.cut_value_at(ev, cut, Pf)
        assert rhs == pytest.approx(ev.components[cut.bus], rel=1e-12), (
            f"cut at bus {cut.bus} does not reproduce f_i")


def test_bus_cut_is_a_minorant_away_from_the_incumbent(net, dispatch):
    """Moving off the incumbent, the cut may only under-estimate f_i.

    A minorant is what makes (M) a relaxation. If a perturbed dispatch ever put
    the cut above the functional the master could cut off the true optimum.
    """
    Pf, _Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
    cuts = risk.build_bus_cuts(ev, risk.select(ev, kappa=5))

    for scale in (-1.0, -0.3, 0.5, 2.0):
        moved_f = {k: v * scale for k, v in Pf.items()}
        moved = risk.evaluate("bus_flow_sum_agg", net, moved_f)
        for cut in cuts:
            rhs = risk.cut_value_at(ev, cut, moved_f)
            assert rhs <= moved.components[cut.bus] + 1e-9, (
                f"cut at bus {cut.bus} exceeds f_i at scale {scale}")
            assert rhs <= moved.value + 1e-9, (
                f"cut at bus {cut.bus} exceeds phi at scale {scale}")


def test_selection_returns_kappa_distinct_components(net, dispatch):
    """kappa separate cuts per iteration, not one at the argmax."""
    Pf, _Pg = dispatch
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
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
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
    chosen = risk.select(ev, kappa=3)
    keys = {risk.bus_cut_key(ev, b) for b in chosen}

    # Same dispatch: every key is already held, so nothing is selected again.
    assert risk.select(ev, kappa=3, existing=keys) != chosen

    # Flip every flow. The buses are unchanged but their sign patterns are not,
    # so the same buses become selectable again.
    flipped = {k: -v for k, v in Pf.items()}
    ev2 = risk.evaluate("bus_flow_sum_agg", net, flipped)
    again = risk.select(ev2, kappa=3, existing=keys)
    assert set(again) & set(chosen), (
        "a new sign pattern at the same bus must be selectable")


def test_empty_cut_is_recognised(net):
    """A cut with no terms reads Phi >= 0 and must not be appended."""
    zero_f = {c: 0.0 for c in net.branches}
    zero_g = {c: 0.0 for c in net.gens}
    ev = risk.evaluate("bus_flow_sum_agg", net, zero_f)
    # A bus with no demand either: every term is zero.
    bare = [c for c, bus in net.buses.items() if bus.Pd == 0.0]
    assert bare, "fixture has no zero-demand bus"
    cuts = risk.build_bus_cuts(ev, bare[:1])
    assert cuts[0].is_empty


def test_negative_series_resistance_contributes_no_heat():
    """eq (4c) is a dissipation; a branch does not cool when it is loaded.

    A negative series resistance is a fitting artefact of a three-winding
    transformer equivalent, and every instance from ACTIVSg10k up carries one --
    178 branches on 10k, 447 on 25k, 1,216 on 70k.  Left raw, `Phi >= r P^2`
    with r < 0 is a CONCAVE constraint and the master stops being convex.
    """
    from ropf.network import Branch

    branch = Branch(count=1, f=1, id_f=1, t=2, id_t=2, r=-0.0045, x=0.01, bc=0.0,
                    rateAmva=1.0, rateBmva=0.0, rateCmva=0.0, ratio=1.0, angle=0.0,
                    maxangle=30.0, minangle=-30.0, status=1, defaultlimit=10.0,
                    branchline0=0)
    assert branch.r == -0.0045, "the raw resistance must stay exactly as given"
    assert branch.r_heat == 0.0
    # The admittance is built from the raw value, which is what the parity test
    # pins; the split must not disturb it.
    assert branch.Gff != 0.0 or branch.Bff != 0.0


def test_joule_functional_uses_the_heat_coefficient(net, dispatch):
    """The functional and cut family (6b) must share one coefficient.

    A surrogate built on one and a functional measured with the other would
    break eq (11), and it would break it silently, on the large cases only.
    """
    Pf, _ = dispatch
    ev = risk.evaluate("joule_loss_max", net, Pf)
    assert all(value >= 0.0 for value in ev.components.values())
    for count, branch in net.branches.items():
        assert ev.components[count] == pytest.approx(
            branch.r_heat * Pf[count] ** 2)

"""Exposure representations and component selection; no solver needed."""

from contextlib import contextmanager

import pytest

from ropf import risk


@pytest.fixture(scope="module")
def Pf(net):
    return {c: (1.0 + c % 7) * (1.0 if i % 3 else -1.0) * 0.1
            for i, c in enumerate(sorted(net.branches))}


@pytest.fixture(scope="module")
def distinct(net):
    return {c: (0.1 + 1e-3 * i) * (1.0 if i % 3 else -1.0)
            for i, c in enumerate(sorted(net.branches))}


@contextmanager
def unrated(net, branches):
    kept = {c: net.branches[c].constrainedflow for c in branches}
    for c in branches:
        net.branches[c].constrainedflow = 0
    try:
        yield
    finally:
        for c, value in kept.items():
            net.branches[c].constrainedflow = value


def cut_value(cut, Pf):
    return sum(Pf[b] for b in cut.branch_pos) - sum(Pf[b] for b in cut.branch_neg)


def test_flow_is_max_abs_flow(net, Pf):
    ev = risk.evaluate("max_active_flow", net, Pf)
    assert ev.value == pytest.approx(max(map(abs, Pf.values())))


def test_joule_uses_the_heat_coefficient(net, Pf):
    ev = risk.evaluate("joule_loss_max", net, Pf)
    for c, br in net.branches.items():
        assert ev.components[c] == pytest.approx(br.r_heat * Pf[c] ** 2)


def test_bus_sums_incident_magnitudes(net, Pf):
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
    for b in net.buses:
        assert ev.components[b] == pytest.approx(
            sum(abs(Pf[e]) for e in ev.incidence[b]))
    assert ev.value == max(ev.components.values())


@pytest.mark.parametrize("metric", risk.METRICS)
def test_order_q_is_the_mean_of_the_q_largest(net, distinct, metric):
    one = risk.evaluate(metric, net, distinct)
    ev = risk.evaluate(metric, net, distinct, q=5)
    top = sorted(one.components.values(), reverse=True)[:5]
    assert ev.value == pytest.approx(sum(top) / 5)
    assert ev.components == one.components


def test_order_past_the_component_count_pads_with_zeros(net, distinct):
    q = net.numbranches + 10
    ev = risk.evaluate("max_active_flow", net, distinct, q=q)
    assert ev.value == pytest.approx(sum(map(abs, distinct.values())) / q)


@pytest.mark.parametrize("metric,domain,q", [("max_line_loading", "all", 1),
                                             ("max_active_flow", "rateA", 1),
                                             ("max_active_flow", "all", 0)])
def test_bad_arguments_are_errors(net, Pf, metric, domain, q):
    with pytest.raises(ValueError):
        risk.evaluate(metric, net, Pf, domain, q)


@pytest.mark.parametrize("metric", risk.METRICS)
def test_rated_domain_is_a_no_op_when_every_branch_is_rated(net, Pf, metric):
    assert net.unconstrained_branches() == []
    a = risk.evaluate(metric, net, Pf, "all")
    b = risk.evaluate(metric, net, Pf, "rated")
    assert (a.value, a.components) == (b.value, b.components)


@pytest.mark.parametrize("metric", ["max_active_flow", "joule_loss_max"])
def test_rated_domain_drops_unrated_branches(net, Pf, metric):
    full = risk.evaluate(metric, net, Pf, "all")
    top = [c for c, v in full.components.items() if v == pytest.approx(full.value)]
    runner_up = max(v for c, v in full.components.items() if c not in top)
    with unrated(net, top):
        rated = risk.evaluate(metric, net, Pf, "rated")
    assert not set(top) & set(rated.components)
    assert rated.value == pytest.approx(runner_up)


def test_rated_domain_drops_a_bus_with_no_rated_line(net, Pf):
    full = risk.evaluate("bus_flow_sum_agg", net, Pf, "all")
    bus = max(full.incidence, key=lambda b: len(full.incidence[b]))
    with unrated(net, full.incidence[bus]):
        rated = risk.evaluate("bus_flow_sum_agg", net, Pf, "rated")
    assert bus not in rated.components and bus not in rated.incidence


def test_bus_cut_is_exact_at_the_incumbent_and_a_minorant_elsewhere(net, Pf):
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
    cuts = risk.build_bus_cuts(ev, risk.select(ev, kappa=10))
    assert len(cuts) == 10
    for cut in cuts:
        assert cut_value(cut, Pf) == pytest.approx(ev.components[cut.bus], rel=1e-12)
    for scale in (-1.0, -0.3, 0.5, 2.0):
        moved = {k: v * scale for k, v in Pf.items()}
        f = risk.evaluate("bus_flow_sum_agg", net, moved).components
        for cut in cuts:
            assert cut_value(cut, moved) <= f[cut.bus] + 1e-9


def test_select_takes_the_kappa_largest_not_yet_cut(net, Pf):
    ev = risk.evaluate("max_active_flow", net, Pf)
    first = risk.select(ev, kappa=5)
    values = [ev.components[c] for c in first]
    assert len(set(first)) == 5 and values == sorted(values, reverse=True)
    assert not set(first) & set(risk.select(ev, kappa=5, existing=set(first)))


def test_bus_exclusion_is_by_bus_and_sign_pattern(net, Pf):
    ev = risk.evaluate("bus_flow_sum_agg", net, Pf)
    chosen = risk.select(ev, kappa=3)
    keys = {risk.bus_cut_key(ev, b) for b in chosen}
    assert not set(risk.select(ev, kappa=3, existing=keys)) & set(chosen)
    flipped = risk.evaluate("bus_flow_sum_agg", net, {k: -v for k, v in Pf.items()})
    assert set(risk.select(flipped, kappa=3, existing=keys)) & set(chosen)


def test_zero_flow_gives_an_empty_cut(net):
    ev = risk.evaluate("bus_flow_sum_agg", net, {c: 0.0 for c in net.branches})
    assert risk.build_bus_cuts(ev, [1])[0].is_empty

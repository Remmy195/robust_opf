"""Section 4: the disfigurements, the frequency screen, and model (D).

The dynamics and disfigurement tests need no solver.  The (D) tests solve on
ACTIVSg200 and skip where no solver is reachable.
"""

from __future__ import annotations

import inspect
import os
import random
import zipfile

import pytest

from ropf.counterfactual import disfigure, dynamics, frequency, postevent
from ropf.counterfactual.frequency import LoadDamping, ScreenConfig
from ropf.counterfactual.postevent import (Disfigurement, PostEventParams,
                                           is_connected, response_window,
                                           survivors)
from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


@pytest.fixture(scope="module")
def net():
    return read_matpower(CASE)


###############################################################################
# The surviving network
###############################################################################


def test_removing_a_bus_removes_its_lines_and_its_units(net):
    """The closure of Section 4.1.2, applied once and in one place."""
    bus = next(c for c, b in net.buses.items()
               if b.genidsbycount and b.degree > 0)
    live_buses, live_branches, live_gens = survivors(
        net, Disfigurement(buses=frozenset({bus})))

    assert bus not in live_buses
    for count in net.buses[bus].genidsbycount:
        assert count not in live_gens
    incident = (set(net.buses[bus].frombranchids.values())
                | set(net.buses[bus].tobranchids.values()))
    assert not (incident & live_branches)


def test_an_offline_unit_is_never_a_survivor(net):
    """It cannot be removed twice, and it cannot contribute capacity."""
    offline = {c for c, g in net.gens.items() if not g.status}
    assert offline, "the fixture has no offline unit; the test proves nothing"
    _, _, live_gens = survivors(net, Disfigurement())
    assert not (offline & live_gens)


def test_connectivity_sees_a_split(net):
    """Section 4.2 excludes a split system before any solve."""
    live_buses, live_branches, _ = survivors(net, Disfigurement())
    assert is_connected(net, live_buses, live_branches)
    # Isolate one bus by cutting every line incident to it.
    bus = next(c for c, b in net.buses.items() if b.degree > 0)
    incident = (set(net.buses[bus].frombranchids.values())
                | set(net.buses[bus].tobranchids.values()))
    assert not is_connected(net, live_buses, live_branches - incident)


###############################################################################
# eq (6d)
###############################################################################


def test_the_response_window_is_the_intersection(net):
    gen = next(iter(net.gens.values()))
    star = 0.5 * (gen.Pmin + gen.Pmax)
    P_star = {gen.count: star}
    lo, hi = response_window(net, P_star, {gen.count: 1.0}, gamma=0.0)
    assert lo[gen.count] == pytest.approx(star)
    assert hi[gen.count] == pytest.approx(star)


def test_the_window_never_leaves_the_operating_range(net):
    """A wide window is clipped by Pmin and Pmax, not by gamma alone."""
    gen = next(iter(net.gens.values()))
    lo, hi = response_window(net, {gen.count: gen.Pmax}, {gen.count: 1.0},
                             gamma=1e6)
    assert lo[gen.count] == pytest.approx(gen.Pmin)
    assert hi[gen.count] == pytest.approx(gen.Pmax)


def test_a_dispatch_a_hair_outside_the_range_does_not_make_d_infeasible(net):
    """P* comes from a solved master and can sit outside by a tolerance.

    Without the guard the naive intersection is empty, and (D) reports a
    contingency for a reason that has nothing to do with the disfigurement.
    """
    gen = next(iter(net.gens.values()))
    outside = gen.Pmax + 1e-9
    lo, hi = response_window(net, {gen.count: outside}, {gen.count: 0.0},
                             gamma=1.0)
    assert lo[gen.count] <= hi[gen.count]
    assert hi[gen.count] == pytest.approx(gen.Pmax)


###############################################################################
# Model (D)
###############################################################################


@pytest.fixture(scope="module")
def evaluator(net):
    pytest.importorskip("amplpy", reason="AMPL is not installed")
    from ropf.model import SolverConfig
    try:
        return postevent.PostEvent(net, SolverConfig(name="gurobi",
                                                     verbose=False))
    except Exception as exc:
        pytest.skip(f"could not load (D): {exc}")


@pytest.fixture(scope="module")
def nominal(net):
    pytest.importorskip("amplpy", reason="AMPL is not installed")
    from ropf.model import Master, SolverConfig
    try:
        master = Master(net, "master.mod", SolverConfig(name="gurobi",
                                                        verbose=False))
        master.set_risk_family("max_active_flow")
        master.set_risk_weight(0.0)
        solution = master.solve()
    except Exception as exc:
        pytest.skip(f"could not solve the master: {exc}")
    if not solution.solved:
        pytest.skip("the master did not solve")
    return solution.Pg


def test_an_undisturbed_system_serves_all_demand(net, evaluator, nominal):
    """(D) at the pre-event dispatch with nothing removed must shed nothing."""
    pi = postevent.capacity_participation(net)
    result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.5),
                                Disfigurement())
    assert result.outcome == "survival"
    assert result.served_all_demand
    assert result.phase == "cost"
    assert result.cost is not None and result.cost > 0


def test_removing_a_load_bus_is_not_reported_as_infeasible(net, evaluator,
                                                           nominal):
    """A removed bus takes its demand with it; the balance row must know that.

    Without `alive_bus` on the demand term the row at a removed load bus reads
    0 = -Pd, so every disfigurement that touched a load bus would come back as
    a system with no feasible post-event dispatch.
    """
    pi = postevent.capacity_participation(net)
    bus = max((c for c, b in net.buses.items() if b.Pd > 0 and b.degree > 1),
              key=lambda c: net.buses[c].Pd)
    result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.5),
                                Disfigurement(buses=frozenset({bus})))
    if result.outcome == "excluded":
        pytest.skip("removing that bus splits the network")
    assert result.outcome == "survival"
    assert result.removed_demand_pu == pytest.approx(net.buses[bus].Pd)
    # eq (6f) sums the shed over the SURVIVING buses, so the demand that left
    # with the bus is not lost load.
    assert bus not in result.shed_by_bus


def test_a_split_network_is_excluded_before_any_solve(net, evaluator, nominal):
    pi = postevent.capacity_participation(net)
    bus = next(c for c, b in net.buses.items() if b.degree > 0)
    incident = (set(net.buses[bus].frombranchids.values())
                | set(net.buses[bus].tobranchids.values()))
    result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.5),
                                Disfigurement(branches=frozenset(incident)))
    assert result.outcome == "excluded"
    assert result.n_solves == 0
    assert result.lost_load is None


def test_the_screen_runs_before_d(net, evaluator, nominal):
    """A collapse never reaches the optimization."""
    pi = postevent.capacity_participation(net)
    result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.5),
                                Disfigurement(),
                                screen=lambda _live: (False, "test collapse"))
    assert result.outcome == "collapse"
    assert result.n_solves == 0
    assert result.cost is None


def test_a_tight_window_forces_load_shedding(net, evaluator, nominal):
    """gamma = 0 pins every unit at P*, so losing one must shed load."""
    pi = postevent.capacity_participation(net)
    biggest = max(nominal, key=lambda g: nominal[g])
    result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.0),
                                Disfigurement(gens=frozenset({biggest})))
    assert result.outcome == "survival"
    assert result.phase == "loadshed"
    assert result.lost_load is not None and result.lost_load > 0
    assert result.lost_load == pytest.approx(sum(result.shed_by_bus.values()),
                                             rel=1e-6)


def test_a_surplus_is_never_put_to_d(net, evaluator, nominal):
    """Removing only demand (no generation) leaves the survivors' own
    pre-event output covering their own demand -- a surplus, curtailed by
    the operator and never handed to the optimization at all.
    """
    pi = postevent.capacity_participation(net)
    candidates = sorted(
        (c for c, b in net.buses.items()
         if b.Pd > 0 and b.degree > 1 and not b.genidsbycount),
        key=lambda c: net.buses[c].degree, reverse=True)
    for bus in candidates:
        result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.5),
                                    Disfigurement(buses=frozenset({bus})))
        if result.outcome != "excluded":
            break
    else:
        pytest.skip("every pure-demand candidate bus splits this fixture")
    assert result.outcome == "survival"
    assert result.lost_load == pytest.approx(0.0)
    assert result.n_solves == 0


def test_evaluations_do_not_leak_into_each_other(net, evaluator, nominal):
    """The campaign reuses one AMPL instance; the state must not accumulate."""
    pi = postevent.capacity_participation(net)
    params = PostEventParams(gamma=0.5)
    first = evaluator.evaluate(nominal, pi, params, Disfigurement())
    biggest = max(nominal, key=lambda g: nominal[g])
    evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.0),
                       Disfigurement(gens=frozenset({biggest})))
    again = evaluator.evaluate(nominal, pi, params, Disfigurement())
    assert again.outcome == first.outcome
    assert again.cost == pytest.approx(first.cost, rel=1e-9)


###############################################################################
# The frequency screen
###############################################################################


def test_the_damping_carries_its_base(net):
    """The textbook figure is on the LOAD base; the swing equation is not."""
    per_load = LoadDamping.per_load(1.5)
    assert per_load.on_system(net) == pytest.approx(1.5 * net.load_pu)
    assert LoadDamping.on_system_base(0.02).on_system(net) == pytest.approx(0.02)


def test_a_bare_number_is_not_a_damping():
    """The failure it prevents is quiet: the nadir just comes out deeper."""
    with pytest.raises(frequency.FrequencyError, match="LOAD base"):
        ScreenConfig(f0_hz=60.0, rocof_max_hz_s=0.5, f_under_hz=59.3,
                     damping=1.5)


def test_the_conversion_factor_is_the_load_ratio():
    """On ACTIVSg2000 the two bases differ by a factor of several hundred."""
    big = os.path.join(os.path.dirname(FIXTURES), "..", "data",
                       "case_ACTIVSg2000.m")
    if not os.path.isfile(big):
        pytest.skip("case_ACTIVSg2000.m not available; unpack it into data/")
    network = read_matpower(big)
    raw, converted = 1.5, LoadDamping.per_load(1.5).on_system(network)
    assert converted / raw > 100, \
        "passing the textbook figure raw would under-damp by this factor"


def test_f_under_above_f0_is_rejected():
    with pytest.raises(frequency.FrequencyError, match="not below"):
        ScreenConfig(f0_hz=60.0, rocof_max_hz_s=0.5, f_under_hz=60.5,
                     damping=LoadDamping.per_load(1.0))


def test_the_frequency_path_has_no_response_window():
    """The screen runs BEFORE (D), so gamma cannot reach the nadir.

    A gamma anywhere in this module would be reporting an effect the method
    does not contain.
    """
    source = inspect.getsource(frequency)
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    body = code.split('"""')
    executable = "".join(body[::2])          # drop the docstrings
    assert "gamma" not in executable.lower(), \
        "gamma reached the frequency screen, which runs before (D)"


def test_rocof_is_the_manuscript_formula(net):
    config = ScreenConfig(f0_hz=60.0, rocof_max_hz_s=1e9, f_under_hz=59.3,
                          damping=LoadDamping.per_load(1.0))
    machines = {c: dynamics.UnitDynamics(bus=g.nodeID, uid="1", H=4.0,
                                         mbase=100.0)
                for c, g in net.gens.items() if g.status}
    live_gens = list(machines)
    P_star = {c: 0.0 for c in live_gens}
    result = frequency.screen(config, net, machines, list(net.buses),
                              live_gens, P_star)
    expected = 60.0 * abs(result.delta_P_pu) / (2.0 * result.H_sys_s)
    assert result.rocof_hz_s == pytest.approx(expected)


def test_a_surplus_produces_no_under_frequency_nadir(net):
    config = ScreenConfig(f0_hz=60.0, rocof_max_hz_s=1e9, f_under_hz=59.3,
                          damping=LoadDamping.per_load(1.0), horizon_s=1.0)
    machines = {c: dynamics.UnitDynamics(bus=g.nodeID, uid="1", H=4.0,
                                         mbase=100.0)
                for c, g in net.gens.items() if g.status}
    nadir, t, n = frequency.integrate_nadir(config, net, machines,
                                            list(machines), {}, -1.0, 10.0)
    assert nadir == config.f0_hz and t == 0.0 and n == 0


def test_no_surviving_inertia_is_a_collapse(net):
    config = ScreenConfig(f0_hz=60.0, rocof_max_hz_s=0.5, f_under_hz=59.3,
                          damping=LoadDamping.per_load(1.0))
    result = frequency.screen(config, net, {}, list(net.buses), [], {})
    assert not result.cleared and result.failed_test == "rocof"


###############################################################################
# .dyr parsing
###############################################################################


def _dyr(tmp_path, text, name="X_dynamics.dyr"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


GENROU = "  101 'GENROU' 1  " + " ".join(["0.1"] * 4 + ["3.75"] + ["0.2"] * 9) + " /"
GENSAL = "  102 'GENSAL' 1  " + " ".join(["0.1"] * 3 + ["2.50"] + ["0.2"] * 8) + " /"
IEEEG1 = ("  101 'IEEEG1' 1  0 0 20.0 " + " ".join(["0.5"] * 19) + " /")


def test_genrou_h_is_the_fifth_parameter(tmp_path):
    units, report = dynamics.parse_dyr(_dyr(tmp_path, GENROU))
    assert units[(101, "1")].H == pytest.approx(3.75)
    assert report.n_rejected == 0


def test_gensal_h_is_the_fourth_parameter(tmp_path):
    units, _ = dynamics.parse_dyr(_dyr(tmp_path, GENSAL))
    assert units[(102, "1")].H == pytest.approx(2.50)


def test_ieeeg1_droop_is_the_reciprocal_of_the_gain(tmp_path):
    """IEEEG1 states a GAIN K.  Using K as the droop overstates stiffness."""
    units, _ = dynamics.parse_dyr(_dyr(tmp_path, IEEEG1))
    assert units[(101, "1")].R == pytest.approx(1.0 / 20.0)


def test_a_record_of_the_wrong_length_is_rejected_not_indexed(tmp_path):
    """Positions are the whole content of the format.

    A record read past a moved field returns a plausible number, and nothing
    downstream would ever notice.  It is counted instead.
    """
    short = "  101 'GENROU' 1  0.1 0.1 0.1 /"
    units, report = dynamics.parse_dyr(_dyr(tmp_path, short))
    assert report.rejected_by_model.get("GENROU") == 1
    assert (101, "1") not in units or units[(101, "1")].H is None
    assert "REJECTED" in report.summary()


def test_an_inverter_carries_no_inertia_and_is_not_a_rejection(tmp_path):
    text = "  103 'REGCA1' 1  0.02 0.02 /"
    units, report = dynamics.parse_dyr(_dyr(tmp_path, text))
    assert units[(103, "1")].H is None
    assert report.n_rejected == 0


###############################################################################
# Distribution irregularities
###############################################################################


def test_locate_finds_the_usual_naming(tmp_path):
    root = tmp_path / "data"
    (root / "extracted").mkdir(parents=True)
    (root / "extracted" / "ACTIVSg2000_dynamics.dyr").write_text(GENROU)
    (root / "extracted" / "ACTIVSg2000.aux").write_text("")
    found = dynamics.locate("ACTIVSg2000", [str(root)])
    assert found.dyr and found.dyr.endswith("ACTIVSg2000_dynamics.dyr")
    assert found.aux


def test_locate_finds_the_activsg25k_naming(tmp_path):
    """ACTIVSg25k drops the `_dynamics` that the other five carry."""
    root = tmp_path / "data"
    (root / "extracted").mkdir(parents=True)
    (root / "extracted" / "ACTIVSg25k.dyr").write_text(GENROU)
    found = dynamics.locate("ACTIVSg25k", [str(root)])
    assert found.dyr and found.dyr.endswith("ACTIVSg25k.dyr")


def test_locate_finds_the_activsg70k_directory(tmp_path):
    """ACTIVSg70k ships unpacked, as a directory, where the others ship a zip."""
    root = tmp_path / "data"
    (root / "ACTIVSg70k").mkdir(parents=True)
    (root / "ACTIVSg70k" / "ACTIVSg70k_dynamics.dyr").write_text(GENROU)
    found = dynamics.locate("ACTIVSg70k", [str(root)])
    assert found.dyr and found.archive is None
    units, _ = dynamics.parse_dyr(found.dyr, found.archive)
    assert units[(101, "1")].H == pytest.approx(3.75)


def test_locate_reads_out_of_a_zip(tmp_path):
    root = tmp_path / "data"
    root.mkdir(parents=True)
    with zipfile.ZipFile(root / "ACTIVSg500.zip", "w") as archive:
        archive.writestr("ACTIVSg500_dynamics.dyr", GENROU)
    found = dynamics.locate("ACTIVSg500", [str(root)])
    assert found.archive and found.dyr
    units, _ = dynamics.parse_dyr(found.dyr, found.archive)
    assert units[(101, "1")].H == pytest.approx(3.75)


def test_locate_reports_nothing_found(tmp_path):
    found = dynamics.locate("ACTIVSg999", [str(tmp_path)])
    assert found.dyr is None and found.aux is None


###############################################################################
# Disfigurements
###############################################################################


def test_the_top_k_ranking_is_deterministic():
    """A tie broken by dictionary order is irreproducible across builds."""
    tied = {3: 1.0, 1: 1.0, 2: 2.0}
    assert disfigure.rank_units(tied) == [2, 1, 3]


def test_the_ranking_is_taken_once_and_held(net, nominal):
    """Section 4.1.1 ranks at lambda = 0 and applies it to every dispatch.

    `top_k_units` takes the ranking, not a dispatch, so a caller cannot pass
    the dispatch under test and re-rank per dispatch by accident -- which would
    flatter the de-risked dispatches, whose top-K carries less.
    """
    parameters = list(inspect.signature(disfigure.top_k_units).parameters)
    assert parameters == ["ranking", "K"]
    ranking = disfigure.rank_units(nominal)
    assert disfigure.top_k_units(ranking, 3).gens == frozenset(ranking[:3])


def test_a_walk_holds_exactly_k_components(net):
    for K in (1, 2, 5, 20, 50):
        walk = disfigure.random_walk(net, K, random.Random(K))
        assert walk.size == K, f"K = {K} gave {walk.size} components"


def test_a_walk_is_connected_in_the_graph(net):
    """The set is a connected subgraph, which is what the class is for."""
    walk = disfigure.random_walk(net, 20, random.Random(3))
    reached = set()
    for count in walk.branches:
        branch = net.branches[count]
        reached.add(branch.id_f)
        reached.add(branch.id_t)
    assert walk.buses <= reached or len(walk.buses) == 1


def test_walks_are_reproducible_from_the_seed(net):
    first = disfigure.walk_draws(net, 10, 5, seed=11)
    second = disfigure.walk_draws(net, 10, 5, seed=11)
    assert [(d.buses, d.branches) for d in first] == \
           [(d.buses, d.branches) for d in second]


def test_different_seeds_give_different_walks(net):
    first = disfigure.walk_draws(net, 20, 5, seed=1)
    second = disfigure.walk_draws(net, 20, 5, seed=2)
    assert [d.buses for d in first] != [d.buses for d in second]


###############################################################################
# The uncurtailable-surplus guard
#
# It used to compare pre-event OUTPUT against surviving demand, which at DC
# balance reduces to "was at least as much load removed as generation".  That is
# true of every branch-only event, where nothing is removed from either side, so
# every branch outage returned "no lost load" without building the flow model.
###############################################################################


def test_a_branch_only_outage_reaches_the_flow_model(net, evaluator, nominal):
    """The regression.  A branch outage removes no load and no generation, so
    the old guard fired on all of them; nothing about serving load says the
    surviving network can still route it within the ratings."""
    pi = postevent.capacity_participation(net)
    params = PostEventParams(gamma=0.5)

    reached = 0
    for branch in sorted(net.branches):
        result = evaluator.evaluate(nominal, pi, params,
                                    Disfigurement(branches=frozenset({branch})))
        if result.outcome != "survival":
            continue
        assert result.phase != "curtailed", (
            f"branch {branch} short-circuited as an uncurtailable surplus, but "
            f"a branch outage removes nothing to curtail")
        if result.n_solves > 0:
            reached += 1
    assert reached > 0, "no branch outage reached (D)"


def test_the_guard_is_on_the_window_floor_not_pre_event_output(net):
    """The floor is what decides it: with no downward room a survivor cannot be
    curtailed at all, and with room to spare it can."""
    pi = postevent.capacity_participation(net)
    live = {c for c, g in net.gens.items() if g.status}

    pinned, _ = response_window(net, {c: net.gens[c].Pmax for c in live},
                                pi, gamma=0.0)
    loose, _ = response_window(net, {c: net.gens[c].Pmax for c in live},
                               pi, gamma=1e6)
    assert sum(pinned[c] for c in live) >= sum(loose[c] for c in live)


def test_an_uncurtailable_surplus_is_marked_curtailed(net, evaluator, nominal):
    """Removing demand with no downward room leaves a surplus (D) cannot
    balance.  It short-circuits, and says so in `phase` so a campaign can count
    the rows that never reached the flow model."""
    pi = postevent.capacity_participation(net)
    candidates = sorted(
        (c for c, b in net.buses.items()
         if b.Pd > 0 and b.degree > 1 and not b.genidsbycount),
        key=lambda c: net.buses[c].Pd, reverse=True)
    for bus in candidates:
        result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.0),
                                    Disfigurement(buses=frozenset({bus})))
        if result.outcome == "survival":
            break
    else:
        pytest.skip("every pure-demand candidate bus splits this fixture")

    assert result.phase == "curtailed"
    assert result.n_solves == 0
    assert result.lost_load == pytest.approx(0.0)


def test_a_wide_window_sends_the_same_event_to_the_model(net, evaluator, nominal):
    """The complement: give the fleet room to ramp down and the same removal is
    no longer uncurtailable, so it goes to (D) instead of short-circuiting."""
    pi = postevent.capacity_participation(net)
    candidates = sorted(
        (c for c, b in net.buses.items()
         if b.Pd > 0 and b.degree > 1 and not b.genidsbycount),
        key=lambda c: net.buses[c].Pd, reverse=True)
    for bus in candidates:
        tight = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.0),
                                   Disfigurement(buses=frozenset({bus})))
        if tight.outcome == "survival" and tight.phase == "curtailed":
            break
    else:
        pytest.skip("no candidate bus short-circuits on this fixture")

    wide = evaluator.evaluate(nominal, pi, PostEventParams(gamma=1e6),
                              Disfigurement(buses=frozenset({bus})))
    assert wide.outcome == "survival"
    assert wide.phase != "curtailed"
    assert wide.n_solves > 0


###############################################################################
# eq (6e) as a slack, and the two paths that report it
#
# `Pf` used to be bounded by beta * U.  At gamma = 0 the response window pins
# Pg at P*, so on a branch outage the post-event flows are DETERMINED, and
# bounding a determined quantity does not measure the violation, it removes the
# solution: both phases returned `infeasible` and the contingencies that
# mattered most reported no number at all.
###############################################################################


def _connected_branches(net, limit=None):
    """Branch outages that leave the network in one piece."""
    live = []
    for count in sorted(net.branches):
        item = Disfigurement(branches=frozenset({count}))
        buses, branches, _ = survivors(net, item)
        if is_connected(net, buses, branches):
            live.append(count)
        if limit is not None and len(live) >= limit:
            break
    return live


def _forced_lp(evaluator, net, P_star, pi, params, item):
    """The same evaluation, forced down the LP even where the closed form runs.

    `evaluate` routes a branch-only event at gamma = 0 to the closed form, so
    the comparison has to reach past it.  `shed_can_move=False` is the gate the
    router would have applied: sum(L) is pinned by the balance identity, so the
    load-shed phase's feasible set is the cost phase's and asking it proves
    nothing.
    """
    buses, branches, gens = survivors(net, item)
    window = response_window(net, P_star, pi, params.gamma)
    counts = dict(n_buses=len(buses), n_branches=len(branches),
                  n_gens=len(gens), removed_demand_pu=0.0)
    return evaluator._solve(params, buses, branches, gens, window, counts,
                            shed_can_move=False)


def test_the_closed_form_and_the_lp_agree(net, evaluator, nominal):
    """The correctness test for the fast path, and the reason it can be
    trusted at 25k and 70k where the LP comparison is not affordable.

    At gamma = 0 on a branch-only event there is no dispatch decision left, so
    the closed form is the exact answer rather than a screen or a bound.  Both
    paths must therefore agree on the severity numbers, not merely bracket each
    other, and nothing downstream may be able to tell which one ran apart from
    `n_solves` and `solve_time_s`.
    """
    pi = postevent.capacity_participation(net)
    params = PostEventParams(gamma=0.0, beta=1.2)

    checked = 0
    for count in _connected_branches(net):
        item = Disfigurement(branches=frozenset({count}))
        fast = evaluator.evaluate(nominal, pi, params, item)
        slow = _forced_lp(evaluator, net, nominal, pi, params, item)

        assert fast.n_solves == 0, "the closed form must not reach AMPL"
        assert slow.n_solves > 0
        assert fast.outcome == slow.outcome == "survival"
        assert fast.phase == slow.phase
        assert fast.worst_loading == pytest.approx(slow.worst_loading,
                                                   rel=1e-8)
        assert fast.overload_max_pu == pytest.approx(slow.overload_max_pu,
                                                     abs=1e-8)
        assert fast.cost == pytest.approx(slow.cost, rel=1e-8)
        assert fast.lost_load == pytest.approx(slow.lost_load, abs=1e-9)
        checked += 1

    assert checked > 0, "no branch outage kept this fixture connected"


def test_the_closed_form_runs_only_where_nothing_may_move(net, evaluator,
                                                          nominal):
    """The routing rule.  Branch-only at gamma = 0 is the closed form; a wider
    window or an event that removes a component is the LP."""
    pi = postevent.capacity_participation(net)
    branch = _connected_branches(net, limit=1)[0]
    item = Disfigurement(branches=frozenset({branch}))

    assert evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.0),
                              item).n_solves == 0
    assert evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.5),
                              item).n_solves > 0

    biggest = max(nominal, key=lambda g: nominal[g])
    unit = evaluator.evaluate(nominal, pi, PostEventParams(gamma=0.0),
                              Disfigurement(gens=frozenset({biggest})))
    assert unit.n_solves > 0


def test_every_admitted_draw_reports_a_loading(net, evaluator, nominal):
    """The point of the slack.  A row that reaches (D) at all must come back
    with a severity number, whichever phase produced it."""
    pi = postevent.capacity_participation(net)
    biggest = max(nominal, key=lambda g: nominal[g])
    branch = _connected_branches(net, limit=1)[0]

    for gamma, item in ((0.0, Disfigurement(branches=frozenset({branch}))),
                        (0.5, Disfigurement(branches=frozenset({branch}))),
                        (0.0, Disfigurement(gens=frozenset({biggest}))),
                        (0.5, Disfigurement())):
        result = evaluator.evaluate(nominal, pi, PostEventParams(gamma=gamma),
                                    item)
        assert result.outcome == "survival"
        assert result.worst_loading is not None, (gamma, result.phase)
        assert result.overload_max_pu is not None
        assert result.overload_sum_pu is not None


def test_a_rating_the_flows_cannot_respect_is_reported_not_refused(
        net, evaluator, nominal):
    """The regression for the whole change.

    beta is bounded below by 1, so the way to make a DETERMINED flow pattern
    violate a rating is to lower the rating.  Halve every rating and the same
    branch outage that cleared at 0.71 of rateA now sits at 1.42 of it, with
    nothing free to move.  Under the hard bound both phases returned
    `infeasible` and the row carried no number.  It must carry a magnitude.
    """
    pi = postevent.capacity_participation(net)
    branch = _connected_branches(net, limit=1)[0]
    item = Disfigurement(branches=frozenset({branch}))
    params = PostEventParams(gamma=0.0, beta=1.0)

    original = {c: b.limit for c, b in net.branches.items()}
    try:
        for count, limit in original.items():
            net.branches[count].limit = 0.5 * limit
        squeezed = postevent.PostEvent(net, evaluator.solver)
        fast = squeezed.evaluate(nominal, pi, params, item)
        slow = _forced_lp(squeezed, net, nominal, pi, params, item)
    finally:
        for count, limit in original.items():
            net.branches[count].limit = limit

    assert fast.phase == "overload"
    assert fast.overload_max_pu > 0.0
    assert fast.overload_sum_pu >= fast.overload_max_pu
    assert fast.worst_loading > 1.0
    # At gamma = 0 on a branch outage sum(L) is pinned by the balance identity,
    # so the overload is the whole of the result and the zero is genuine.
    assert fast.lost_load == pytest.approx(0.0)

    assert slow.phase == "overload"
    assert slow.overload_max_pu == pytest.approx(fast.overload_max_pu, rel=1e-6)
    assert slow.worst_loading == pytest.approx(fast.worst_loading, rel=1e-8)


###############################################################################
# The tolerances of Section 3.6
###############################################################################


def test_the_tolerances_scale_with_the_fleet(net):
    """They used to be absolute and to sit at the solver's own feasibility
    tolerance, so a fleet-wide sum was tested against the slack the solver is
    entitled to leave on one row of it.  On ACTIVSg2000 that decided three
    weights of eighteen."""
    assert postevent.shed_tolerance(net) == pytest.approx(
        postevent.SHED_RTOL * net.load_pu)
    assert postevent.shed_tolerance(net) > postevent.SHED_RTOL
    assert postevent.tolerance(1e-6, 0.1) == pytest.approx(1e-6)


def test_a_branch_only_event_never_reaches_the_surplus_guard(net, evaluator,
                                                             nominal):
    """Item 2.4.  A branch outage removes nothing from either side, so the
    guard has no surplus to find and can only misfire -- which it did, taking
    the whole branch class with it whenever the master's balance residual
    landed on the far side of an absolute tolerance."""
    pi = postevent.capacity_participation(net)
    for count in _connected_branches(net):
        item = Disfigurement(branches=frozenset({count}))
        for gamma in (0.0, 0.5):
            result = evaluator.evaluate(nominal, pi,
                                        PostEventParams(gamma=gamma), item)
            assert result.phase != "curtailed", (count, gamma)

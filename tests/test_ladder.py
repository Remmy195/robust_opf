"""The ladder driver: the grid it enumerates, and the claim that makes it safe.

The claim is the part worth testing hardest.  It is what lets the study be run
in several shells at once and resumed after a crash, and both of those are
silently wrong if two processes can hold the same combo -- the second would
overwrite the first's artifacts with a half-finished campaign and the DONE
marker would say it was fine.
"""

from __future__ import annotations

import os

import pytest

from ropf.config import ConfigError
from ropf.risk import METRICS
from ropf.study import ladder
from ropf.study.ladder import Combo, LadderConfig, read_ladder_config


def base(**kwargs):
    """A config that passes validation, so a test can vary one thing."""
    defaults = {"damping_per_load": 1.0}
    defaults.update(kwargs)
    return LadderConfig(**defaults)


def write(tmp_path, text, name="ladder.conf"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


###############################################################################
# The key table
###############################################################################


def test_the_two_refusals_are_inherited(tmp_path):
    """Unknown and duplicate keys are errors here for the same reason they are
    in a run config -- and by the same code, not a second copy of it."""
    with pytest.raises(ConfigError, match="unknown key 'rung'"):
        read_ladder_config(write(tmp_path, "rung = activs200\n"))
    with pytest.raises(ConfigError, match="already set on line 1"):
        read_ladder_config(write(tmp_path, "k_bar = 5\nk_bar = 9\n"))


def test_a_config_file_round_trips(tmp_path):
    path = write(tmp_path, "rungs = activs200, texas2k\n"
                           "metrics = max_active_flow\n"
                           "stages = baseline\n"
                           "gamma = 0.25, 0.5\n"
                           "damping_per_load = 1.5\n")
    config = read_ladder_config(path)
    assert config.rungs == ("activs200", "texas2k")
    assert config.gamma == (0.25, 0.5)
    assert config.damping_per_load == 1.5


def test_an_unknown_rung_is_named(tmp_path):
    with pytest.raises(ConfigError, match="activs300"):
        base(rungs=("activs300",))


def test_an_unknown_metric_is_rejected():
    with pytest.raises(ConfigError, match="joule"):
        base(metrics=("joule",))


def test_an_unknown_stage_is_rejected():
    with pytest.raises(ConfigError, match="a4"):
        base(stages=("a4",))


def test_an_empty_axis_is_an_error():
    with pytest.raises(ConfigError, match="at least one"):
        base(metrics=())


###############################################################################
# The damping base
###############################################################################


def test_a_damping_without_a_base_is_refused():
    """Neither given: the screen would have no damping at all."""
    with pytest.raises(ConfigError, match="exactly one"):
        LadderConfig()


def test_two_damping_bases_are_refused():
    """Both given: which one applies has no defensible answer, and the failure
    is quiet -- the nadir just comes out at the wrong depth."""
    with pytest.raises(ConfigError, match="exactly one"):
        LadderConfig(damping_per_load=1.0, damping_on_system=0.02)


def test_the_damping_carries_its_base():
    assert base(damping_per_load=1.5).damping().base == "load"
    per_system = LadderConfig(damping_on_system=0.02)
    assert per_system.damping().base == "system"


###############################################################################
# The screen constants
###############################################################################


def test_the_screen_refuses_to_invent_its_constants():
    """f0, RoCoF-bar and f_under are grid-code quantities. A config that omits
    them must stop, not supply a plausible number that gets reported as data."""
    with pytest.raises(ConfigError) as exc:
        base().screen_config()
    message = str(exc.value)
    for name in ("f0_hz", "rocof_max_hz_s", "f_under_hz"):
        assert name in message
    assert "campaign = false" in message, "the error must name the way out"


def test_a_complete_screen_config_builds():
    config = base(f0_hz=60.0, rocof_max_hz_s=0.5, f_under_hz=59.3)
    screen = config.screen_config()
    assert screen.f0_hz == 60.0
    assert screen.f_under_hz == 59.3


###############################################################################
# Combos
###############################################################################


def test_the_full_grid_is_six_by_three_by_three():
    assert len(ladder.combos(base())) == 54


def test_combos_run_the_cheap_rungs_first_whatever_the_file_says():
    """A partial run must cover the cheap rungs first, so the order is the
    ladder's own and not the order the config happens to list."""
    config = base(rungs=("activs70k", "activs200", "texas2k"),
                  metrics=("max_active_flow",), stages=("baseline",))
    assert [c.rung for c in ladder.combos(config)] == [
        "activs200", "texas2k", "activs70k"]


def test_a_combo_names_its_distribution_not_its_rung():
    """`locate` wants ACTIVSg2000; the rung is called texas2k."""
    assert Combo("texas2k", "max_active_flow", "baseline").distribution \
        == "ACTIVSg2000"
    assert Combo("activs200", "max_active_flow", "baseline").name \
        == "activs200_max_active_flow_baseline"


def test_the_evaluation_count_is_the_declared_grid():
    """6 weights x 1 gamma x (8 top-K + 3 x 100 walks)."""
    assert base().evaluations_per_combo() == 1848
    assert base(gamma=(0.25, 0.5, 1.0)).evaluations_per_combo() == 5544
    assert base(campaign=False).evaluations_per_combo() == 0


def test_a_combo_becomes_a_run_config():
    config = base(kappa=3, eta=0.9, k_bar=7, weight_grid=(0.0, 1.0))
    combo = Combo("activs200", "joule_loss_max", "a2")
    run = config.run_config(combo, "/tmp/case.m")
    assert (run.metric, run.stage) == ("joule_loss_max", "a2")
    assert (run.kappa, run.eta, run.k_bar) == (3, 0.9, 7)
    assert run.tag == combo.name


def test_the_thread_count_reaches_gurobi():
    """A thread per logical cpu cost 5.7x on the top rung; the declared count
    has to actually arrive at the solver, not just at Knitro."""
    solver = base(threads=16).postevent_solver()
    assert solver.gurobi_threads == 16
    assert solver.knitro_threads == 16


###############################################################################
# Claiming
###############################################################################


def test_a_combo_can_only_be_claimed_once(tmp_path):
    directory = str(tmp_path / "combo")
    assert ladder.claim(directory) is True
    assert ladder.claim(directory) is False, \
        "a second process must not get the same combo"


def test_a_claim_records_who_took_it(tmp_path):
    directory = str(tmp_path / "combo")
    ladder.claim(directory)
    assert os.path.isfile(os.path.join(directory, ladder.CLAIM, "who"))


def test_a_released_claim_can_be_retaken(tmp_path):
    directory = str(tmp_path / "combo")
    ladder.claim(directory)
    ladder.release(directory)
    assert ladder.claim(directory) is True


def test_done_is_what_finished_means(tmp_path):
    directory = str(tmp_path / "combo")
    ladder.claim(directory)
    assert not ladder.is_done(directory), "a claim is not a completion"
    open(os.path.join(directory, ladder.DONE), "w").close()
    assert ladder.is_done(directory)


def test_next_combo_skips_the_finished_and_the_claimed(tmp_path):
    config = base(rungs=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline", "a2", "a3"), outdir=str(tmp_path))
    first = ladder.next_combo(config)
    second = ladder.next_combo(config)
    assert first is not None and second is not None
    assert first != second, "the claim on the first must push the second on"

    # finish both, and the third is the only one left
    for combo in (first, second):
        open(os.path.join(ladder.combo_dir(config, combo), ladder.DONE),
             "w").close()
    third = ladder.next_combo(config)
    assert third not in (first, second)
    assert ladder.next_combo(config) is None, "nothing should be left"


def test_a_stale_claim_is_released_only_when_it_is_stale(tmp_path):
    config = base(rungs=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline",), outdir=str(tmp_path),
                  stale_claim_hours=24.0)
    combo = ladder.next_combo(config)
    directory = ladder.combo_dir(config, combo)

    assert ladder.reclaim_stale(config) == [], "a fresh claim is not stale"

    old = os.path.join(directory, ladder.CLAIM)
    os.utime(old, (0, 0))          # claimed at the epoch
    assert ladder.reclaim_stale(config) == [combo]
    assert ladder.claim(directory) is True


def test_a_finished_combo_is_never_reclaimed(tmp_path):
    """A DONE combo with an old claim left behind must stay done."""
    config = base(rungs=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline",), outdir=str(tmp_path))
    combo = ladder.next_combo(config)
    directory = ladder.combo_dir(config, combo)
    open(os.path.join(directory, ladder.DONE), "w").close()
    os.utime(os.path.join(directory, ladder.CLAIM), (0, 0))
    assert ladder.reclaim_stale(config) == []


def test_status_reports_all_three_states(tmp_path):
    config = base(rungs=("activs200",), metrics=("max_active_flow",),
                  stages=("baseline", "a2", "a3"), outdir=str(tmp_path))
    done = ladder.next_combo(config)
    open(os.path.join(ladder.combo_dir(config, done), ladder.DONE), "w").close()
    ladder.next_combo(config)          # leaves one running

    states = {row["combo"]: row["state"]
              for row in ladder.status_rows(config)}
    assert sorted(states.values()) == ["done", "pending", "running"]

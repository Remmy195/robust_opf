"""AGC participation: ``GenParFac`` intersected with ``GenAGCAble``.

The distribution states the factor and the AGC-able flag separately and they
disagree, so the two are read separately and the flag wins.  A unit that is not
AGC-able does not respond to an event whatever factor sits beside it.
"""

from __future__ import annotations

import os

import pytest

from ropf.counterfactual import dynamics
from ropf.counterfactual.dynamics import UnitParticipation
from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")
AUX = os.path.join(DATA, "ACTIVSg200.aux")


def _aux_or_skip() -> str:
    if not os.path.exists(AUX):
        pytest.skip(f"{AUX} not found")
    return AUX


@pytest.fixture(scope="module")
def network():
    return read_matpower(CASE)


###############################################################################
# The tokenizer, which is what hid the flag
###############################################################################


def test_the_tokenizer_keeps_a_double_quoted_field_with_a_space_whole():
    """``GenAGCAble`` is written ``"NO "``.  Splitting it shifts every later
    field on the row by one and the row then fails to parse, which silently
    dropped every non-AGC unit."""
    row = '71 "1" "Closed" 1.04 71 100.0 "NO " 4.0 1.23'
    tokens = dynamics._TOKEN.findall(row)
    assert tokens[6] == '"NO "'
    assert float(tokens[7].strip('"')) == 4.0


def test_the_tokenizer_still_keeps_single_quoted_fields_whole():
    """PSS/E quotes with ``'``; the .dyr reader depends on it."""
    tokens = dynamics._TOKEN.findall("1 'GENROU' 1 8.0 0.05")
    assert tokens[1] == "'GENROU'"


###############################################################################
# Parsing
###############################################################################


def test_parse_reads_a_row_for_every_unit(network):
    aux = dynamics.parse_participation(_aux_or_skip())
    assert len(aux) == network.numgens


def test_parse_keeps_the_factor_and_the_flag_apart():
    aux = dynamics.parse_participation(_aux_or_skip())
    assert all(isinstance(u, UnitParticipation) for u in aux.values())
    # The fixture instance has non-AGC units, and their factor is preserved as
    # written rather than overwritten with zero.
    assert any(not u.agc_able for u in aux.values())


def test_effective_is_zero_exactly_when_the_unit_is_not_agc_able():
    assert UnitParticipation(factor=4.0, agc_able=False).effective == 0.0
    assert UnitParticipation(factor=4.0, agc_able=True).effective == 4.0
    assert UnitParticipation(factor=0.0, agc_able=True).effective == 0.0


def test_a_file_with_no_agc_column_is_taken_as_all_able(tmp_path):
    aux = tmp_path / "noflag.aux"
    aux.write_text('DATA (Gen, [BusNum,GenID,GenParFac])\n{\n'
                   ' 1 "1" 10.0\n 2 "1" 5.0\n}\n')
    parsed = dynamics.parse_participation(str(aux))
    assert len(parsed) == 2
    assert all(u.agc_able for u in parsed.values())


###############################################################################
# The join
###############################################################################


def test_no_non_agc_unit_receives_participation(network):
    aux = dynamics.parse_participation(_aux_or_skip())
    factors, source = dynamics.participation_factors(network, aux)
    assert source == "aux_genparfac_agc"

    for (bus, uid), unit in aux.items():
        if unit.agc_able:
            continue
        siblings = [c for c in sorted(network.gens)
                    if int(network.gens[c].nodeID) == bus]
        uids = sorted(u for b, u in aux if b == bus)
        if len(siblings) != len(uids):
            continue                            # bus took the capacity split
        count = siblings[uids.index(uid)]
        assert factors[count] == 0.0


def test_no_out_of_service_unit_receives_participation(network):
    aux = dynamics.parse_participation(_aux_or_skip())
    factors, _ = dynamics.participation_factors(network, aux)
    for count, gen in network.gens.items():
        if not gen.status:
            assert factors[count] == 0.0


def test_the_factors_normalize_to_one(network):
    aux = dynamics.parse_participation(_aux_or_skip())
    factors, _ = dynamics.participation_factors(network, aux)
    assert sum(factors.values()) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in factors.values())


def test_every_generator_has_a_factor(network):
    aux = dynamics.parse_participation(_aux_or_skip())
    factors, _ = dynamics.participation_factors(network, aux)
    assert set(factors) == set(network.gens)


def test_an_empty_aux_falls_back_to_the_capacity_surrogate(network):
    factors, source = dynamics.participation_factors(network, {})
    assert source == "capacity_surrogate"
    assert sum(factors.values()) == pytest.approx(1.0)


def test_an_aux_matching_nothing_falls_back_to_the_surrogate(network):
    absent = {(10 ** 9, "1"): UnitParticipation(factor=1.0, agc_able=True)}
    factors, source = dynamics.participation_factors(network, absent)
    assert source == "capacity_surrogate"


def test_an_all_non_agc_fleet_raises_rather_than_substituting(network):
    """The one substitution this function must never make.  Falling back to the
    capacity surrogate here would make the whole fleet responsive on the
    strength of a file that says none of it is."""
    dead = {}
    for count in sorted(network.gens):
        bus = int(network.gens[count].nodeID)
        uid = str(sum(1 for c in sorted(network.gens)
                      if int(network.gens[c].nodeID) == bus and c <= count))
        dead[(bus, uid)] = UnitParticipation(factor=5.0, agc_able=False)

    with pytest.raises(ValueError, match="every effective"):
        dynamics.participation_factors(network, dead)


def test_the_capacity_split_only_shares_agc_able_weight(network):
    """The fallback path, exercised directly: a bus whose unit count disagrees
    with the .aux must not spread weight from a unit the file says is off AGC."""
    bus = int(network.gens[min(network.gens)].nodeID)
    # One .aux entry too many for the bus, so the exact join cannot apply.
    n = sum(1 for c in network.gens if int(network.gens[c].nodeID) == bus)
    mixed = {(bus, str(i)): UnitParticipation(factor=10.0, agc_able=False)
             for i in range(1, n + 2)}
    with pytest.raises(ValueError, match="every effective"):
        dynamics.participation_factors(network, mixed)

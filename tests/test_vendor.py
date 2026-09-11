"""Class (c): the vendor N-1 list.

The fixture is `case_ACTIVSg200.m`, the one case tracked in the repository, and
`ACTIVSg200.aux` beside it where the distribution has been unpacked.  Tests that
need the .aux skip when it is absent so the suite still runs on a bare clone.
"""

from __future__ import annotations

import os

import pytest

from ropf.counterfactual import vendor
from ropf.counterfactual.postevent import Disfigurement, survivors
from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")
AUX = os.path.join(DATA, "ACTIVSg200.aux")


def _aux_or_skip() -> str:
    if not os.path.exists(AUX):
        pytest.skip(f"{AUX} not unpacked; see README")
    return AUX


@pytest.fixture(scope="module")
def network():
    return read_matpower(CASE)


###############################################################################
# Parsing
###############################################################################


def test_element_parses_the_three_kinds():
    assert vendor._element("BRANCH 1001 1064 2", "OPEN") == vendor.Element(
        "BRANCH", (1001, 1064), "2", "OPEN")
    assert vendor._element("GEN 1004 1", "OPEN") == vendor.Element(
        "GEN", (1004,), "1", "OPEN")
    assert vendor._element("SHUNT 1007 1", "OPEN") == vendor.Element(
        "SHUNT", (1007,), "1", "OPEN")


def test_element_rejects_what_it_does_not_understand():
    assert vendor._element("", "OPEN") is None
    assert vendor._element("INTERFACE 3", "OPEN") is None
    assert vendor._element("BRANCH 1001 1064", "OPEN") is None
    assert vendor._element("BRANCH a b 1", "OPEN") is None


def test_parse_aux_reads_single_element_contingencies():
    contingencies = vendor.parse_aux(_aux_or_skip())
    assert contingencies
    # GO3 Eq. (231): exactly one outaged element per contingency.
    assert all(len(c.elements) == 1 for c in contingencies)
    assert all(c.label for c in contingencies)


def test_parse_aux_is_empty_when_the_case_ships_no_list(tmp_path):
    empty = tmp_path / "none.aux"
    empty.write_text("DATA (Bus, [BusNum])\n{\n 1\n}\n")
    assert vendor.parse_aux(str(empty)) == []


def test_parse_aux_raises_on_a_missing_file():
    with pytest.raises(FileNotFoundError):
        vendor.parse_aux(os.path.join(DATA, "does_not_exist.aux"))


###############################################################################
# Mapping into the count space
###############################################################################


def test_every_mapped_contingency_matches_the_element_it_names(network):
    contingencies = vendor.parse_aux(_aux_or_skip())
    by_label = {c.label: c for c in contingencies}
    mapped, _ = vendor.map_contingencies(network, contingencies)
    assert mapped

    for item in mapped:
        source = by_label[item.label].elements[0]
        if item.branches:
            branch = network.branches[next(iter(item.branches))]
            assert {int(branch.f), int(branch.t)} == set(source.buses)
        else:
            unit = network.gens[next(iter(item.gens))]
            assert int(unit.nodeID) == source.buses[0]


def test_a_mapped_contingency_names_exactly_one_component(network):
    mapped, _ = vendor.map_contingencies(
        network, vendor.parse_aux(_aux_or_skip()))
    for item in mapped:
        assert len(item.branches) + len(item.gens) + len(item.buses) == 1


def test_labels_are_unique(network):
    mapped, _ = vendor.map_contingencies(
        network, vendor.parse_aux(_aux_or_skip()))
    labels = [item.label for item in mapped]
    assert len(labels) == len(set(labels))


def test_the_report_accounts_for_every_contingency(network):
    contingencies = vendor.parse_aux(_aux_or_skip())
    _, report = vendor.map_contingencies(network, contingencies)
    assert report.total == len(contingencies)
    assert (report.mapped + report.skipped_shunt + report.skipped_multi
            + report.unmapped_branch + report.unmapped_gen
            + report.unmapped_other) == report.total


def test_parallel_circuits_resolve_to_different_branches(network):
    """The whole point of matching by position.  A pair whose two circuits both
    mapped to the same count would outage one line twice and never say so."""
    contingencies = vendor.parse_aux(_aux_or_skip())
    by_label = {c.label: c for c in contingencies}
    mapped, _ = vendor.map_contingencies(network, contingencies)

    seen = {}
    for item in mapped:
        if not item.branches:
            continue
        element = by_label[item.label].elements[0]
        key = (min(element.buses), max(element.buses))
        seen.setdefault(key, []).append(next(iter(item.branches)))
    for key, counts in seen.items():
        assert len(counts) == len(set(counts)), f"circuits collided on {key}"


def test_an_unresolvable_circuit_is_reported_not_guessed(network):
    """A pair the .aux names two circuits for but the case file carries one row
    for must not resolve.  Guessing would outage a different line than the one
    the vendor named."""
    branch = network.branches[min(network.branches)]
    f, t = int(branch.f), int(branch.t)
    pair = (min(f, t), max(f, t))
    parallel = sum(1 for b in network.branches.values()
                   if (min(int(b.f), int(b.t)), max(int(b.f), int(b.t))) == pair)

    invented = [
        vendor.VendorContingency(
            label=f"INVENTED_C{n}",
            elements=(vendor.Element("BRANCH", (f, t), str(n), "OPEN"),))
        for n in range(1, parallel + 2)          # one more circuit than rows
    ]
    mapped, report = vendor.map_contingencies(network, invented)
    assert mapped == []
    assert report.unmapped_branch == len(invented)


def test_a_multi_element_contingency_is_skipped(network):
    pair = vendor.VendorContingency(
        label="TWO", elements=(vendor.Element("BRANCH", (1, 2), "1", "OPEN"),
                               vendor.Element("BRANCH", (2, 3), "1", "OPEN")))
    mapped, report = vendor.map_contingencies(network, [pair])
    assert mapped == []
    assert report.skipped_multi == 1


def test_a_non_open_action_is_skipped(network):
    closed = vendor.VendorContingency(
        label="CLOSE", elements=(vendor.Element("BRANCH", (1, 2), "1", "CLOSE"),))
    mapped, report = vendor.map_contingencies(network, [closed])
    assert mapped == []
    assert report.skipped_multi == 1


def test_shunt_contingencies_are_counted_not_dropped(network):
    shunt = vendor.VendorContingency(
        label="S1", elements=(vendor.Element("SHUNT", (1,), "1", "OPEN"),))
    mapped, report = vendor.map_contingencies(network, [shunt])
    assert mapped == []
    assert report.skipped_shunt == 1


###############################################################################
# What the campaign does with them
###############################################################################


def test_mapped_contingencies_feed_the_removal_closure(network):
    """A vendor Disfigurement has to be a Disfigurement: `survivors` must accept
    it and remove exactly what it names."""
    mapped, _ = vendor.map_contingencies(
        network, vendor.parse_aux(_aux_or_skip()))
    branch_case = next(item for item in mapped if item.branches)

    live_buses, live_branches, _ = survivors(network, branch_case)
    assert len(live_buses) == network.numbuses
    assert len(live_branches) == network.numbranches - 1
    assert next(iter(branch_case.branches)) not in live_branches


def test_vendor_draws_is_empty_for_a_case_with_no_list(network, tmp_path):
    empty = tmp_path / "none.aux"
    empty.write_text("DATA (Bus, [BusNum])\n{\n 1\n}\n")
    mapped, report = vendor.vendor_draws(network, str(empty))
    assert mapped == []
    assert report.total == 0


###############################################################################
# Three-winding transformers, and where circuit identity comes from
###############################################################################


def test_element_parses_a_three_winding_transformer():
    assert vendor._element("3WXFORMER 10065 10066 10067 1", "OPEN") == vendor.Element(
        vendor.XFMR3, (10065, 10066, 10067), "1", "OPEN")
    assert vendor._element("3WXFORMER 10065 10066 1", "OPEN") is None


def test_three_winding_contingencies_are_counted_not_dropped(network):
    """The regression this class was built to prevent, and did not.  A
    ``3WXFORMER`` used to fall through `_element`, so `parse_aux` dropped the
    contingency before `MappingReport.total` was taken and the logged coverage
    was a rate over an already-truncated list."""
    three = vendor.VendorContingency(
        label="3WT_1",
        elements=(vendor.Element(vendor.XFMR3, (1, 2, 3), "1", "OPEN"),))
    mapped, report = vendor.map_contingencies(network, [three])
    assert mapped == []
    assert report.total == 1
    assert report.skipped_3w == 1
    assert (report.mapped + report.skipped_shunt + report.skipped_3w
            + report.skipped_multi + report.unmapped_branch
            + report.unmapped_gen + report.unmapped_other) == report.total


def test_parse_aux_keeps_every_label_the_file_carries():
    """`parse_aux` must not lose a contingency on the way in.  Counted against
    the raw ``CTGLabel`` column so the test cannot agree with the bug."""
    path = _aux_or_skip()
    text = open(path, encoding="utf-8", errors="replace").read()
    fields, rows = vendor._aux_block(text, "ContingencyElement")
    at = fields.index("CTGLabel")
    raw = {vendor._TOKENS.findall(r.strip())[at].strip('"')
           for r in rows if len(vendor._TOKENS.findall(r.strip())) > at}
    assert {c.label for c in vendor.parse_aux(path)} == raw


def test_aux_blocks_finds_every_branch_block():
    """The .aux writes Branch as two blocks -- lines and transformers.  A reader
    that stops at the first sees no transformer at all."""
    text = open(_aux_or_skip(), encoding="utf-8", errors="replace").read()
    blocks = vendor._aux_blocks(text, "Branch")
    assert len(blocks) >= 2
    assert any("LineX" in fields for fields, _ in blocks)
    assert any("LinePhase" in fields for fields, _ in blocks)


def test_circuits_come_from_the_aux_branch_table_not_the_contingency_list(network):
    """A pair carrying two circuits but contingency-listing one must still
    resolve.  Reading identity off the contingency list made that a length
    disagreement and `_resolve` refused the pair."""
    pair_at = vendor.circuits_from_aux(_aux_or_skip())
    assert pair_at

    counts_at = vendor._branch_index(network)
    for pair, idents in pair_at.items():
        if pair in counts_at:
            assert len(idents) == len(counts_at[pair]), pair


def test_an_unlisted_parallel_circuit_no_longer_blocks_its_sibling():
    """The ACTIVSg500 failure, reduced to the dicts that caused it.  The fixture
    case has no parallel pair, so this is built by hand rather than found: two
    circuits in the network, one of them contingency-listed.  Reading identity
    off the contingency list makes the lengths disagree and loses BOTH; reading
    it off the .aux Branch table resolves the listed one to its own count."""
    counts_at = {(14, 386): [101, 102]}
    element = vendor.Element("BRANCH", (14, 386), "1", "OPEN")

    from_list = {(14, 386): ["1"]}                  # only circuit 1 is listed
    assert vendor._resolve(element, counts_at, from_list, keyed_by_pair=True) is None

    from_aux = {(14, 386): ["1", "2"]}              # the case carries both
    assert vendor._resolve(element, counts_at, from_aux, keyed_by_pair=True) == 101
    sibling = vendor.Element("BRANCH", (14, 386), "2", "OPEN")
    assert vendor._resolve(sibling, counts_at, from_aux, keyed_by_pair=True) == 102


###############################################################################
# Device type in the circuit order
#
# Cross-checked 2026-09-07 against ground truth the case distributions carry:
# `ACTIVSg####.con` is PowerWorld's own contingency export and names
# (from bus, to bus, circuit) directly, and `contab_ACTIVSg####.m` is built from
# that .con by MATPOWER's PSSECON2CHGTAB, with `label` the 1-based position of
# the contingency in the .con and `row` the mpc.branch row.  Joining the two
# resolves every branch contingency without guessing.  It found four
# disagreements on ACTIVSg2000, all on the one bus pair carrying both a line and
# a transformer, and none anywhere else across the six cases.
###############################################################################


def test_branch_records_carry_the_device_type():
    record = vendor.branch_records_from_aux(_aux_or_skip())
    assert record
    flat = [entry for entries in record.values() for entry in entries]
    assert all(isinstance(ident, str) and isinstance(is_xfmr, bool)
               for ident, is_xfmr in flat)
    # The fixture case has both kinds; a reader seeing one block would not.
    assert any(is_xfmr for _, is_xfmr in flat)
    assert any(not is_xfmr for _, is_xfmr in flat)


def test_circuits_from_aux_is_the_identifier_projection():
    record = vendor.branch_records_from_aux(_aux_or_skip())
    idents = vendor.circuits_from_aux(_aux_or_skip())
    assert idents == {pair: [ident for ident, _ in entries]
                      for pair, entries in record.items()}


def test_branch_types_line_up_with_branch_counts(network):
    counts = vendor._branch_index(network)
    types = vendor._branch_types(network)
    assert set(counts) == set(types)
    assert all(len(counts[pair]) == len(types[pair]) for pair in counts)


def test_a_mixed_pair_is_reordered_into_the_case_file_order():
    """The ACTIVSg2000 defect, reduced to the dicts that caused it.

    Pair (7161, 7292) carries three lines with circuits 2, 3, 4 and one
    two-winding transformer with circuit 1.  The .aux writes its Branch records
    as a line block then a transformer block, so it yields ['2', '3', '4', '1'];
    mpc.branch holds them transformer first at rows 2620 to 2623.  Zipping the
    two as read sent circuit '1' to row 2623 and shifted the three lines down.
    """
    record = {(7161, 7292): [("2", False), ("3", False), ("4", False),
                             ("1", True)]}
    counts = {(7161, 7292): [2620, 2621, 2622, 2623]}
    types = {(7161, 7292): [True, False, False, False]}

    unaligned = {pair: [i for i, _ in v] for pair, v in record.items()}
    transformer = vendor.Element("BRANCH", (7161, 7292), "1", "OPEN")
    assert vendor._resolve(transformer, counts, unaligned,
                           keyed_by_pair=True) == 2623        # the defect

    aligned = vendor._align_to_network(record, types)
    assert aligned == {(7161, 7292): ["1", "2", "3", "4"]}
    assert vendor._resolve(transformer, counts, aligned,
                           keyed_by_pair=True) == 2620
    for ident, row in zip(("2", "3", "4"), (2621, 2622, 2623)):
        line = vendor.Element("BRANCH", (7161, 7292), ident, "OPEN")
        assert vendor._resolve(line, counts, aligned, keyed_by_pair=True) == row


def test_alignment_leaves_a_single_type_pair_alone():
    """Every pair on five of the six cases is all lines or all transformers, so
    the reordering has to be the identity there."""
    record = {(1, 2): [("1", False), ("2", False)],
              (3, 4): [("1", True)]}
    types = {(1, 2): [False, False], (3, 4): [True]}
    assert vendor._align_to_network(record, types) == {(1, 2): ["1", "2"],
                                                       (3, 4): ["1"]}


def test_alignment_drops_a_pair_whose_types_disagree():
    """Equal lengths but different device types is an inconsistency between the
    .aux and the case file.  Dropping the pair makes `_resolve` refuse it and
    the report count it, which beats outaging a component nobody named."""
    record = {(1, 2): [("1", False), ("2", False)]}
    types = {(1, 2): [True, False]}
    assert vendor._align_to_network(record, types) == {}
    element = vendor.Element("BRANCH", (1, 2), "1", "OPEN")
    assert vendor._resolve(element, {(1, 2): [7, 8]},
                           vendor._align_to_network(record, types),
                           keyed_by_pair=True) is None


def test_alignment_leaves_a_length_disagreement_to_resolve():
    """An out-of-service parallel gives a shorter count list.  `_resolve` already
    refuses that, so the reordering must not swallow it first."""
    record = {(1, 2): [("1", False), ("2", False)]}
    types = {(1, 2): [False]}
    assert vendor._align_to_network(record, types) == {(1, 2): ["1", "2"]}


def test_vendor_draws_resolves_the_mixed_pair_on_activsg2000():
    """The defect in place, on the case that carries it.  ACTIVSg200 has no pair
    holding both a line and a transformer, so this needs the larger case."""
    case = os.path.join(DATA, "case_ACTIVSg2000.m")
    aux = os.path.join(DATA, "ACTIVSg2000.aux")
    for path in (case, aux):
        if not os.path.exists(path):
            pytest.skip(f"{path} not unpacked; see README")

    net = read_matpower(case)
    mapped, _ = vendor.vendor_draws(net, aux)
    contingencies = {c.label: c for c in vendor.parse_aux(aux)}

    seen = {}
    for item in mapped:
        element = contingencies[item.label].elements[0]
        if element.kind == vendor.BRANCH and set(element.buses) == {7161, 7292}:
            seen[element.ident] = (item.label, next(iter(item.branches)))

    # The mpc.branch rows contab_ACTIVSg2000.m gives for these four circuits.
    assert {ident: row for ident, (_, row) in seen.items()} == {
        "1": 2620, "2": 2621, "3": 2622, "4": 2623}
    # Circuit 1 is the transformer, which is what the .aux ordering misplaced.
    assert seen["1"][0].startswith("T_")
    assert all(seen[ident][0].startswith("L_") for ident in ("2", "3", "4"))

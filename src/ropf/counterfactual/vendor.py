"""Class (c): the vendor N-1 list shipped with the case.

Five of the six ACTIVSg instances carry an enumerated single-element contingency
list in their PowerWorld ``.aux``, as ``Contingency`` and ``ContingencyElement``
records.  This reads it into `Disfigurement` objects.

WHY THE CLASS EXISTS.  The sampled classes give a worst case that is an order
statistic, biased optimistic by an amount depending on the draw count.  This one
is ENUMERATED, so its worst case is exact -- which makes it their control -- and
it is a list the case authors wrote rather than one this study constructed.

STRUCTURE MATCHES GO3 Eq. (231): each contingency names exactly one element.
Unlike Eq. (230) the elements are not restricted to branches.

NOT REPRESENTABLE.  A shunt outage changes ``Gs``, which is a bus-level
aggregate here and not a switchable per-shunt quantity.  A three-winding
transformer is three ``TransformerWinding`` records at a star bus, so it is not
a single-element contingency.  Both are counted and skipped, never dropped:
until 2026-09-06 a ``3WXFORMER`` fell through `_element` before
`MappingReport.total` was taken, and the logged coverage was a rate over a list
already truncated -- "11806/11806" on ACTIVSg10k, which holds 12,106.

CIRCUIT IDENTIFIERS.  MATPOWER carries none, so a parallel pair cannot be
matched to ``BRANCH f t 1`` / ``BRANCH f t 2`` by name.  It is matched by
POSITION against the ``.aux`` own ``Branch`` table.  Where the two lists differ
in length the pair is reported unmapped rather than guessed at: guessing would
outage a circuit the vendor did not name and nothing downstream would notice.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..network import Network
from . import dynamics
from .postevent import Disfigurement

#: Element kinds the ``.aux`` uses.  Anything outside this set is reported.
BRANCH, GEN, SHUNT = "BRANCH", "GEN", "SHUNT"
#: A three-winding transformer.  Named here so it is COUNTED rather than
#: dropped: it used to fall through `_element` as unrecognized, which removed
#: the contingency before `MappingReport.total` was taken and made the reported
#: coverage a rate over a list that had already been silently truncated --
#: 300 of 12,106 on ACTIVSg10k and 750 of 26,655 on ACTIVSg25k.
XFMR3 = "3WXFORMER"

_TOKENS = re.compile(r'"[^"]*"|\S+')


def _noop(_message: str) -> None:
    return None


###############################################################################
# What the file says
###############################################################################


@dataclass(frozen=True)
class Element:
    """One outaged element, exactly as the ``.aux`` names it."""

    kind: str
    #: BRANCH: (from bus, to bus) as FILE bus IDs.  GEN/SHUNT: (bus, bus).
    buses: Tuple[int, ...]
    #: Circuit or unit identifier, as a string.  MATPOWER has no equivalent.
    ident: str
    action: str


@dataclass(frozen=True)
class VendorContingency:
    """One contingency: a label and the elements it outages."""

    label: str
    elements: Tuple[Element, ...]


@dataclass
class MappingReport:
    """What survived the translation into `Disfigurement`, and what did not.

    Every count here is reported by the campaign.  A class that quietly drops a
    third of the vendor list and reports a rate over what remains is measuring
    something the reader cannot name.
    """

    total: int = 0
    mapped: int = 0
    skipped_shunt: int = 0
    #: Three-winding transformer contingencies.  Representable as a Disfigurement
    #: -- each one is exactly three ``TransformerWinding`` records meeting at a
    #: star bus -- but not as a SINGLE-element one, so it is out of the GO3
    #: Eq. (231) scope this class is built to.  Counted, not mapped.
    skipped_3w: int = 0
    skipped_multi: int = 0
    unmapped_branch: int = 0
    unmapped_gen: int = 0
    unmapped_other: int = 0
    #: Labels of the contingencies that did not map, for spot-checking.
    examples: List[str] = field(default_factory=list)

    def note(self, label: str) -> None:
        if len(self.examples) < 10:
            self.examples.append(label)

    @property
    def skipped(self) -> int:
        return self.total - self.mapped

    def summary(self) -> str:
        return (f"{self.mapped}/{self.total} vendor contingencies mapped "
                f"(skipped: {self.skipped_shunt} shunt, "
                f"{self.skipped_3w} three-winding transformer, "
                f"{self.skipped_multi} multi-element, "
                f"{self.unmapped_branch} branch not in service, "
                f"{self.unmapped_gen} unit not in service, "
                f"{self.unmapped_other} other)")


###############################################################################
# Reading the .aux
###############################################################################


def parse_aux(path: str,
              archive: Optional[str] = None,
              log: Optional[Callable[[str], None]] = None
              ) -> List[VendorContingency]:
    """Read the ``Contingency`` and ``ContingencyElement`` records of an ``.aux``.

    Reads through `dynamics` so a packed ``.aux`` is read the same way here as
    for ``GenParFac``.  Returns empty rather than raising when the file has no
    contingency block: ACTIVSg70k ships none, which is a fact about the
    distribution.  The caller reports the emptiness.
    """
    emit = log or _noop
    if archive is None and not os.path.exists(path):
        raise FileNotFoundError(f"aux file not found: {path}")

    text = dynamics._read(path, archive)

    fields, rows = _aux_block(text, "ContingencyElement")
    if fields is None:
        emit(f" vendor list: {os.path.basename(path)} carries no "
             f"ContingencyElement block\n")
        return []

    try:
        i_object = fields.index("Object")
        i_label = fields.index("CTGLabel")
        i_action = fields.index("Action")
    except ValueError:
        emit(f" vendor list: ContingencyElement block lacks Object/CTGLabel/"
             f"Action; fields were {fields}\n")
        return []

    by_label: Dict[str, List[Element]] = {}
    order: List[str] = []
    for row in rows:
        token = _TOKENS.findall(row.strip())
        if len(token) <= max(i_object, i_label, i_action):
            continue
        element = _element(token[i_object].strip('"'),
                           token[i_action].strip('"').strip())
        if element is None:
            continue
        label = token[i_label].strip('"')
        if label not in by_label:
            by_label[label] = []
            order.append(label)
        by_label[label].append(element)

    emit(f" vendor list: {len(order)} contingencies, "
         f"{sum(len(v) for v in by_label.values())} elements, from "
         f"{os.path.basename(path)}\n")
    return [VendorContingency(label=label, elements=tuple(by_label[label]))
            for label in order]


def _element(object_field: str, action: str) -> Optional[Element]:
    """``"BRANCH 1001 1064 1"`` -> an `Element`.  None when unrecognized."""
    part = object_field.split()
    if not part:
        return None
    kind = part[0].upper()
    try:
        if kind == BRANCH and len(part) >= 4:
            return Element(BRANCH, (int(part[1]), int(part[2])), part[3], action)
        if kind == XFMR3 and len(part) >= 5:
            return Element(XFMR3, (int(part[1]), int(part[2]), int(part[3])),
                           part[4], action)
        if kind in (GEN, SHUNT) and len(part) >= 3:
            return Element(kind, (int(part[1]),), part[2], action)
    except ValueError:
        return None
    return None


def _aux_blocks(text: str, object_name: str
                ) -> List[Tuple[List[str], List[str]]]:
    """EVERY ``DATA (<object>, [...])`` block, as (field list, body rows).

    Plural on purpose.  The ``.aux`` writes ``Branch`` as TWO blocks with
    different field lists -- lines carry ``LineR``/``LineX``/``LineC``,
    transformers carry ``LineXFType``/``XFTapMax``/``LinePhase`` -- so a reader
    that stops at the first one silently sees no transformer at all.
    """
    out: List[Tuple[List[str], List[str]]] = []
    for match in re.finditer(r"^DATA \(" + re.escape(object_name) +
                             r",\s*\[(.*?)\]\s*\)\s*\n\{(.*?)\n\}",
                             text, re.S | re.M):
        fields = [f.strip() for f in
                  re.split(r",\s*", match.group(1).replace("\n", " ")) if f.strip()]
        out.append((fields, match.group(2).strip().splitlines()))
    return out


def _aux_block(text: str, object_name: str
               ) -> Tuple[Optional[List[str]], List[str]]:
    """The field list and body rows of the FIRST ``DATA (<object>, [...])`` block.

    Correct for the single-block objects this module reads by name
    (``ContingencyElement``, ``Gen``).  For ``Branch`` use `_aux_blocks`.
    """
    blocks = _aux_blocks(text, object_name)
    if not blocks:
        return None, []
    return blocks[0]


def branch_records_from_aux(path: str,
                            archive: Optional[str] = None
                            ) -> Dict[Tuple[int, int], List[Tuple[str, bool]]]:
    """Unordered bus pair -> ``(circuit identifier, is transformer)``, in file order.

    THE DEVICE TYPE IS PART OF THE IDENTITY.  The ``.aux`` writes ``Branch`` as a
    line block then a transformer block; ``mpc.branch`` does not always agree --
    on ACTIVSg2000 the pair (7161, 7292) has three lines and one transformer,
    ordered transformer first, and zipping the lists sent all four of its
    contingencies one row off.  `_align_to_network` uses this flag to restore
    the case file order.

    ``BranchDeviceType`` carries the type where the ``.aux`` writes it; the
    block field list is the fallback, since only the transformer block carries
    the ``XF...`` and ``LinePhase`` columns.
    """
    text = dynamics._read(path, archive)
    seen: Dict[Tuple[int, int], List[Tuple[str, bool]]] = {}
    for fields, rows in _aux_blocks(text, "Branch"):
        try:
            i_f = fields.index("BusNum")
            i_t = fields.index("BusNum:1")
            i_c = fields.index("LineCircuit")
        except ValueError:
            continue
        i_s = fields.index("LineStatus") if "LineStatus" in fields else None
        i_d = (fields.index("BranchDeviceType")
               if "BranchDeviceType" in fields else None)
        block_is_xfmr = any(f.startswith("XF") or f == "LinePhase" for f in fields)
        for row in rows:
            token = _TOKENS.findall(row.strip())
            if len(token) < len(fields):
                continue
            if i_s is not None and not token[i_s].strip('"').strip(
                    ).lower().startswith("closed"):
                continue
            try:
                f, t = int(token[i_f].strip('"')), int(token[i_t].strip('"'))
            except ValueError:
                continue
            if i_d is not None:
                is_xfmr = token[i_d].strip('"').strip().lower().startswith(
                    "transformer")
            else:
                is_xfmr = block_is_xfmr
            seen.setdefault((min(f, t), max(f, t)), []).append(
                (token[i_c].strip('"').strip(), is_xfmr))
    return seen


def circuits_from_aux(path: str,
                      archive: Optional[str] = None
                      ) -> Dict[Tuple[int, int], List[str]]:
    """Unordered bus pair -> circuit identifiers, from the ``.aux`` ``Branch`` table.

    THIS IS THE RIGHT SOURCE OF CIRCUIT IDENTITY, and `_circuits` is not: that
    one reads identifiers off the CONTINGENCY LIST, so a pair with two circuits
    in the case but named once in the N-1 list gave two counts against one
    identifier and `_resolve` refused the pair -- every unmapped branch
    contingency on ACTIVSg500.

    The identifiers come back in the ``.aux`` order, which is NOT always the
    order `_branch_index` produces counts in; comparing ``LineAMVA`` against
    ``rateA`` cannot see a permutation among records of equal rating.  Callers
    that can permute should read `branch_records_from_aux`.

    Out-of-service records are dropped: `ropf.network` does not carry them.
    """
    return {pair: [ident for ident, _ in record]
            for pair, record in branch_records_from_aux(path, archive).items()}


###############################################################################
# Translating into the count space
###############################################################################


def _branch_index(network: Network) -> Dict[Tuple[int, int], List[int]]:
    """Unordered file bus pair -> branch counts, in file order.

    Out-of-service branches are not in ``network.branches`` at all, so a pair
    can hold fewer entries here than the ``.aux`` names circuits for.  That is
    exactly the case `map_contingencies` refuses to resolve.
    """
    index: Dict[Tuple[int, int], List[int]] = {}
    for count in sorted(network.branches):
        branch = network.branches[count]
        key = (min(int(branch.f), int(branch.t)),
               max(int(branch.f), int(branch.t)))
        index.setdefault(key, []).append(int(count))
    return index


def _branch_types(network: Network) -> Dict[Tuple[int, int], List[bool]]:
    """Unordered file bus pair -> is-transformer flags, in the same order as
    `_branch_index` produces counts.

    ``Branch.is_transformer`` is the case file's own tap column before
    `ropf.network` normalizes a tap of 0 to 1, so it distinguishes a transformer
    from a line exactly as MATPOWER's ``makeYbus`` does.
    """
    index: Dict[Tuple[int, int], List[bool]] = {}
    for count in sorted(network.branches):
        branch = network.branches[count]
        key = (min(int(branch.f), int(branch.t)),
               max(int(branch.f), int(branch.t)))
        index.setdefault(key, []).append(bool(branch.is_transformer))
    return index


def _align_to_network(record: Dict[Tuple[int, int], List[Tuple[str, bool]]],
                      branch_types: Dict[Tuple[int, int], List[bool]]
                      ) -> Dict[Tuple[int, int], List[str]]:
    """``.aux`` circuit identifiers, reordered into the case file own order.

    `_resolve` matches by POSITION, so the lists must agree.  The ``.aux`` groups
    lines then transformers and ``mpc.branch`` interleaves, so only a mixed pair
    changes here; the reordering walks the case device-type sequence and takes
    the next ``.aux`` identifier of that type.

    A pair whose type multisets disagree is DROPPED, leaving the contingency
    unmapped and reported -- the standing choice over outaging a component the
    vendor did not name.
    """
    aligned: Dict[Tuple[int, int], List[str]] = {}
    for pair, entries in record.items():
        idents = [ident for ident, _ in entries]
        want = branch_types.get(pair)
        if want is None or len(want) != len(entries):
            # `_resolve` refuses a length disagreement on its own.
            aligned[pair] = idents
            continue
        pool: Dict[bool, List[str]] = {True: [], False: []}
        for ident, is_xfmr in entries:
            pool[is_xfmr].append(ident)
        if len(pool[True]) != sum(want):
            continue
        aligned[pair] = [pool[flag].pop(0) for flag in want]
    return aligned


def _gen_index(network: Network) -> Dict[int, List[int]]:
    """File bus ID -> generator counts at that bus, in file order."""
    index: Dict[int, List[int]] = {}
    for count in sorted(network.gens):
        index.setdefault(int(network.gens[count].nodeID), []).append(int(count))
    return index


def _circuits(contingencies: Sequence[VendorContingency]
              ) -> Dict[Tuple[int, int], List[str]]:
    """Unordered bus pair -> the circuit identifiers the CONTINGENCY LIST names.

    The fallback for a caller with no ``.aux`` path.  Prefer `circuits_from_aux`:
    this one sees only the circuits that happen to be contingency-listed, so a
    pair whose second circuit is not listed looks like a length disagreement and
    `_resolve` refuses it.
    """
    seen: Dict[Tuple[int, int], set] = {}
    for contingency in contingencies:
        for element in contingency.elements:
            if element.kind != BRANCH or len(element.buses) != 2:
                continue
            key = (min(element.buses), max(element.buses))
            seen.setdefault(key, set()).add(element.ident)
    return {key: sorted(value) for key, value in seen.items()}


def _units(contingencies: Sequence[VendorContingency]) -> Dict[int, List[str]]:
    """Bus -> the unit identifiers the ``.aux`` names at it."""
    seen: Dict[int, set] = {}
    for contingency in contingencies:
        for element in contingency.elements:
            if element.kind != GEN:
                continue
            seen.setdefault(element.buses[0], set()).add(element.ident)
    return {key: sorted(value) for key, value in seen.items()}


def map_contingencies(network: Network,
                      contingencies: Sequence[VendorContingency],
                      log: Optional[Callable[[str], None]] = None,
                      circuit_at: Optional[Dict[Tuple[int, int], List[str]]] = None,
                      circuit_record: Optional[
                          Dict[Tuple[int, int], List[Tuple[str, bool]]]] = None
                      ) -> Tuple[List[Disfigurement], MappingReport]:
    """Translate the vendor list into `Disfigurement` objects.

    Only single-element ``OPEN`` contingencies on branches and units are
    mapped.  Everything else is counted in the report and skipped.  The label
    carried through is the vendor's own ``CTGLabel``, so a row in the campaign
    CSV can be traced back to the ``.aux``.

    ``circuit_record`` is the pair -> ``(identifier, is transformer)`` index from
    `branch_records_from_aux`, and is what `vendor_draws` supplies; it is
    reordered into the case file's own order by `_align_to_network` before any
    branch is resolved.  ``circuit_at`` is the same index without the device
    type, used as given.  Passing neither falls back to the contingency list, so
    a caller holding no ``.aux`` path (the tests, which invent contingencies
    against a fixture) behaves as before.  See `circuits_from_aux` for why the
    contingency list is the wrong source when an ``.aux`` is available.
    """
    emit = log or _noop
    report = MappingReport(total=len(contingencies))

    branch_at = _branch_index(network)
    gen_at = _gen_index(network)
    if circuit_record is not None:
        circuit_at = _align_to_network(circuit_record, _branch_types(network))
    elif circuit_at is None:
        circuit_at = _circuits(contingencies)
    unit_at = _units(contingencies)

    mapped: List[Disfigurement] = []
    for contingency in contingencies:
        element = _single_open(contingency)
        if element is None:
            report.skipped_multi += 1
            report.note(contingency.label)
            continue

        if element.kind == SHUNT:
            report.skipped_shunt += 1
            continue

        if element.kind == XFMR3:
            report.skipped_3w += 1
            continue

        if element.kind == BRANCH:
            count = _resolve(element, branch_at, circuit_at, keyed_by_pair=True)
            if count is None:
                report.unmapped_branch += 1
                report.note(contingency.label)
                continue
            mapped.append(Disfigurement(branches=frozenset({count}),
                                        label=contingency.label))
            report.mapped += 1
            continue

        if element.kind == GEN:
            count = _resolve(element, gen_at, unit_at, keyed_by_pair=False)
            if count is None:
                report.unmapped_gen += 1
                report.note(contingency.label)
                continue
            mapped.append(Disfigurement(gens=frozenset({count}),
                                        label=contingency.label))
            report.mapped += 1
            continue

        report.unmapped_other += 1
        report.note(contingency.label)

    emit(f" {report.summary()}\n")
    return mapped, report


def _single_open(contingency: VendorContingency) -> Optional[Element]:
    """The one element of a single-element OPEN contingency, else None."""
    if len(contingency.elements) != 1:
        return None
    element = contingency.elements[0]
    return element if element.action.upper() == "OPEN" else None


def _resolve(element: Element,
             counts_at: Dict,
             idents_at: Dict,
             keyed_by_pair: bool) -> Optional[int]:
    """Match a named circuit or unit to a count by position.

    The two lists -- the counts the case file carries and the identifiers the
    ``.aux`` names -- are matched only when they are the same length.  A pair
    with an out-of-service parallel gives a shorter count list, and rather than
    outaging the wrong circuit the caller is told it did not resolve.
    """
    key = ((min(element.buses), max(element.buses))
           if keyed_by_pair else element.buses[0])
    counts = counts_at.get(key)
    idents = idents_at.get(key)
    if not counts or not idents or len(counts) != len(idents):
        return None
    try:
        return counts[idents.index(element.ident)]
    except ValueError:
        return None


###############################################################################
# The entry point the campaign calls
###############################################################################


def vendor_draws(network: Network,
                 aux_path: str,
                 archive: Optional[str] = None,
                 log: Optional[Callable[[str], None]] = None
                 ) -> Tuple[List[Disfigurement], MappingReport]:
    """Class (c) for one case: parse, then map.  Empty when the case ships none.

    Circuit identity comes from the same ``.aux``'s ``Branch`` table and not from
    its contingency list; `circuits_from_aux` says why.  The device type comes
    with it, so that a bus pair carrying both a line and a transformer is
    resolved in the case file's order; `branch_records_from_aux` says why.
    """
    contingencies = parse_aux(aux_path, archive, log)
    if not contingencies:
        return [], MappingReport(total=0)
    return map_contingencies(
        network, contingencies, log,
        circuit_record=branch_records_from_aux(aux_path, archive))

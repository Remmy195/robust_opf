"""Machine data for the frequency screen: inertia, droop, and participation.

Three quantities come out of the PowerWorld/PSS/E distributions that ship with
the ACTIVSg cases and are not in the MATPOWER ``.m`` file at all:

    H     inertia constant, seconds on the machine's own MVA base (``.dyr``)
    R, T  droop and the dominant governor lag (``.dyr``)
    pi_g  AGC participation factor (``.aux``, field ``GenParFac``)

EVERY RECORD IS CHECKED AGAINST ITS DECLARED LAYOUT BEFORE IT IS INDEXED.  A
PSS/E ``.dyr`` record is positional -- ``H`` is the 5th GENROU parameter and the
4th of a GENSAL -- so the wrong position yields a plausible number, not an
error.  `parse_dyr` requires each record to carry exactly the parameter count
its model declares and REJECTS it otherwise, counting the rejections, so a moved
layout shows up as a count rather than a screen that passes everything.

THE DISTRIBUTIONS ARE NOT UNIFORM and the exceptions are not guessable:
ACTIVSg25k names its file ``ACTIVSg25k.dyr`` with no ``_dynamics``, and
ACTIVSg70k ships unpacked rather than as a ``.zip``.  `locate` knows both.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..network import Network

#: Machine models, as (parameters after the machine ID, {field: index}).
#: Positions follow the ANDES PSS/E record layout (andes/io/psse-dyr.yaml).
MACHINE_MODELS: Dict[str, Tuple[int, Dict[str, int]]] = {
    # T'do T''do T'qo T''qo H D Xd Xq X'd X'q X''d Xl S(1.0) S(1.2)
    "GENROU": (14, {"H": 4}),
    # T'do T''do T''qo H D Xd Xq X'd X''d Xl S(1.0) S(1.2)
    "GENSAL": (12, {"H": 3}),
}

#: Governor models.  IEEEG1 states a GAIN K, and the droop is 1/K -- not K.
GOVERNOR_MODELS: Dict[str, Tuple[int, Dict[str, int]]] = {
    # Rselect Fswitch R Tpelec maxerr minerr Kpgov KIgov Kdgov Tdgov VMAX VMIN
    # Tact Kturb Wfnl Tb Tc Teng Tfload Kpload KIload Ldref Dm Ropen Rclose
    # KImw Aset Ka Ta Trate db Tsa Tsb Rup Rdown
    "GGOV1": (35, {"R": 2, "T": 15}),
    # JBUS M K T1 T2 T3 Uo Uc PMAX PMIN T4 K1 K2 T5 K3 K4 T6 K5 K6 T7 K7 K8
    "IEEEG1": (22, {"K": 2, "T": 5}),
    # R r Tr Tf Tg VELM Gmax Gmin Twater At Dturb qNL
    "HYGOV": (12, {"R": 0, "T": 8}),
    # R T1 VMAX VMIN T2 T3 Dt.  T3 is the reheat lag and is the one that shapes
    # the ramp (6.3 to 9 s on ACTIVSg200, against T1 of 0.3 to 0.5 s).
    # Without this entry the two smallest instances integrate with NO primary
    # response at all -- 49 of 49 units on ACTIVSg200 are governed by TGOV1 and
    # nothing else -- and every nadir there comes out far too deep.
    "TGOV1": (7, {"R": 0, "T": 5}),
    # R T1 T2 T3 AT KT VMAX VMIN Dturb.  T1 is the fuel-valve lag and is the
    # ramp: T3 is the load-limiter path, which is a temperature limit, not the
    # base primary response.  30 of the 51 units on ACTIVSg500 are GAST.
    "GAST": (9, {"R": 0, "T": 1}),
}

#: Inverter-interfaced machines.  They carry no ``H``, and that is not missing
#: data: a converter contributes no synchronous inertia to the COI.
INVERTER_MODELS = frozenset({"REGCA1", "REECA1", "REPCA1", "REGCAU1", "WT3G1",
                             "WT4G1", "PVGU1", "PVEU1"})

#: One field of a ``.dyr`` record or an ``.aux`` row.  BOTH quote styles are
#: recognized: PSS/E quotes with ``'``, PowerWorld with ``"``.  Handling only
#: the single quote splits a PowerWorld field that contains a space -- the
#: ``GenAGCAble`` value is written ``"NO "`` with a trailing space inside the
#: quotes -- into two tokens, which shifts every later field on that row by one
#: and made the row fail to parse.  That silently dropped every non-AGC unit:
#: 39 of 90 on ACTIVSg500, 128 of 544 on ACTIVSg2000, 1349 of 2485 on
#: ACTIVSg10k.
_TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def _noop(_message: str) -> None:
    return None


class DynamicsError(ValueError):
    """The dynamics data is not something the screen will silently accept."""


###############################################################################
# Records
###############################################################################


@dataclass
class UnitDynamics:
    """What one machine contributes to the COI response."""

    bus: int
    uid: str
    #: Inertia constant, s on `mbase`.  None for an inverter or an unmodelled
    #: unit, which then contributes no inertia -- correct, not missing.
    H: Optional[float] = None
    #: The machine's own MVA base, from the ``.RAW``.  Without it H cannot be
    #: put on the system base.
    mbase: Optional[float] = None
    #: Droop, p.u.  None when the unit has no governor and so no primary
    #: response.
    R: Optional[float] = None
    #: The dominant governor/turbine lag, s.
    T: Optional[float] = None
    machine_model: str = ""
    governor_model: str = ""

    @property
    def is_responsive(self) -> bool:
        return bool(self.R and self.T and self.R > 0 and self.T > 0)


@dataclass
class DynamicsReport:
    """What the parse saw, so a layout change is a number and not a silence."""

    path: str = ""
    records_by_model: Dict[str, int] = field(default_factory=dict)
    rejected_by_model: Dict[str, int] = field(default_factory=dict)
    n_units: int = 0
    n_with_inertia: int = 0
    n_responsive: int = 0

    @property
    def n_rejected(self) -> int:
        return sum(self.rejected_by_model.values())

    def summary(self) -> str:
        parts = [f"{self.n_units} machines, {self.n_with_inertia} with inertia,"
                 f" {self.n_responsive} with primary response"]
        if self.rejected_by_model:
            parts.append("REJECTED for a parameter count that disagrees with "
                         "the declared layout: "
                         + ", ".join(f"{m} x{n}" for m, n
                                     in sorted(self.rejected_by_model.items())))
        return "; ".join(parts)


###############################################################################
# Locating the distribution
###############################################################################


@dataclass(frozen=True)
class DynamicsFiles:
    """Where a instance's dynamics data actually is."""

    dyr: Optional[str] = None
    aux: Optional[str] = None
    #: Set when the files are inside a zip, which the readers open directly.
    archive: Optional[str] = None


def locate(instance: str, search: Sequence[str]) -> DynamicsFiles:
    """Find the ``.dyr`` and ``.aux`` for a instance, across both distribution shapes.

    `instance` is the distribution name, e.g. ``ACTIVSg25k``.  Both irregularities
    are handled here so no caller has to know them: ACTIVSg25k drops the
    ``_dynamics`` from its ``.dyr`` name, and ACTIVSg70k ships as a directory
    rather than a zip.
    """
    dyr_names = (f"{instance}_dynamics.dyr", f"{instance}.dyr")
    aux_names = (f"{instance}.aux", f"{instance}_dynamics.aux")

    for root in search:
        if not os.path.isdir(root):
            continue
        # Unpacked, either loose in the directory or in a subdirectory named
        # for the instance -- the ACTIVSg70k shape.
        for directory in (root, os.path.join(root, instance),
                          os.path.join(root, "extracted")):
            if not os.path.isdir(directory):
                continue
            dyr = _first_existing(directory, dyr_names)
            aux = _first_existing(directory, aux_names)
            if dyr or aux:
                return DynamicsFiles(dyr=dyr, aux=aux)

        archive = os.path.join(root, f"{instance}.zip")
        if os.path.isfile(archive):
            with zipfile.ZipFile(archive) as handle:
                held = {os.path.basename(n): n for n in handle.namelist()}
            dyr = next((held[n] for n in dyr_names if n in held), None)
            aux = next((held[n] for n in aux_names if n in held), None)
            if dyr or aux:
                return DynamicsFiles(dyr=dyr, aux=aux, archive=archive)

    return DynamicsFiles()


def _first_existing(directory: str, names: Iterable[str]) -> Optional[str]:
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def _read(path: str, archive: Optional[str] = None) -> str:
    if archive:
        with zipfile.ZipFile(archive) as handle:
            return handle.read(path).decode("utf-8", errors="replace")
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


###############################################################################
# .dyr
###############################################################################


def parse_dyr(path: str,
              archive: Optional[str] = None,
              log: Optional[Callable[[str], None]] = None
              ) -> Tuple[Dict[Tuple[int, str], UnitDynamics], DynamicsReport]:
    """Read a PSS/E ``.dyr`` into per-unit inertia, droop and governor lag.

    A record is ``bus MODEL id p1 p2 ...`` terminated by ``/``.  A record whose
    parameter count disagrees with its model's declared layout is REJECTED, not
    indexed into: the positions are the whole content of the format, so reading
    past a moved field returns a plausible number and nothing else would ever
    notice.  Rejections are counted per model in the report.
    """
    emit = log or _noop
    text = _read(path, archive)
    units: Dict[Tuple[int, str], UnitDynamics] = {}
    report = DynamicsReport(path=(f"{archive}:{path}" if archive else path))

    for record in text.split("/"):
        record = record.strip()
        if not record:
            continue
        tokens = _TOKEN.findall(record.replace("\n", " "))
        if len(tokens) < 3:
            continue
        try:
            bus = int(tokens[0])
        except ValueError:
            continue

        model = tokens[1].strip("'").strip().upper()
        uid = tokens[2].strip("'").strip()
        params = tokens[3:]
        report.records_by_model[model] = report.records_by_model.get(model, 0) + 1

        if model in INVERTER_MODELS:
            # No H by construction, so nothing to read and nothing to reject.
            units.setdefault((bus, uid), UnitDynamics(bus=bus, uid=uid))
            continue

        layout = MACHINE_MODELS.get(model) or GOVERNOR_MODELS.get(model)
        if layout is None:
            continue                      # exciter, PSS, stabilizer: not ours

        expected, index = layout
        if len(params) != expected:
            report.rejected_by_model[model] = \
                report.rejected_by_model.get(model, 0) + 1
            continue

        unit = units.setdefault((bus, uid), UnitDynamics(bus=bus, uid=uid))
        try:
            if model in MACHINE_MODELS:
                unit.H = float(params[index["H"]])
                unit.machine_model = model
            else:
                unit.governor_model = model
                if "K" in index:
                    # IEEEG1 states a GAIN.  The droop is its reciprocal; using
                    # K directly would understate the droop by K^2 and make
                    # every unit look far stiffer than it is.
                    gain = float(params[index["K"]])
                    unit.R = (1.0 / gain) if gain > 0 else None
                else:
                    droop = float(params[index["R"]])
                    unit.R = droop if droop > 0 else None
                lag = float(params[index["T"]])
                unit.T = lag if lag > 0 else None
        except (ValueError, IndexError):
            report.rejected_by_model[model] = \
                report.rejected_by_model.get(model, 0) + 1

    report.n_units = len(units)
    report.n_with_inertia = sum(1 for u in units.values() if u.H)
    report.n_responsive = sum(1 for u in units.values() if u.is_responsive)
    emit(f" dynamics: {report.summary()}\n")
    return units, report


###############################################################################
# .aux participation factors
###############################################################################

_GENPARFAC = re.compile(r"GenParFac", re.IGNORECASE)
_GENAGCABLE = re.compile(r"GenAGCAble", re.IGNORECASE)


@dataclass(frozen=True)
class UnitParticipation:
    """One unit's AGC data, as the ``.aux`` states it.

    ``factor`` and ``agc_able`` are kept apart rather than folded into a single
    number because the distribution states them apart and they disagree.  On
    ACTIVSg2000, 128 of 544 units are ``GenAGCAble = NO`` and still carry a
    nonzero ``GenParFac``; on ACTIVSg10k it is 1349 of 2485.  A unit that is not
    AGC-able does not respond, whatever factor sits beside it, so folding the
    flag in at parse time would hide the disagreement instead of recording it.
    """

    factor: float
    agc_able: bool

    @property
    def effective(self) -> float:
        """The factor that actually applies.  Zero for a unit not on AGC."""
        return self.factor if self.agc_able else 0.0


def parse_participation(path: str,
                        archive: Optional[str] = None,
                        log: Optional[Callable[[str], None]] = None
                        ) -> Dict[Tuple[int, str], UnitParticipation]:
    """pi_g from the PowerWorld ``.aux`` ``GenParFac`` and ``GenAGCAble`` fields.

    Returns ``{(bus, unit id): UnitParticipation}``, unnormalized -- PowerWorld
    states participation on an arbitrary scale, and the caller normalizes over
    whatever fleet survives, which is not the fleet the file was written for.

    ``GenAGCAble`` is read here and applied by `participation_factors`.  A file
    with no such column is treated as all-AGC-able, which is what the field's
    absence meant before it existed, and the log says so.

    Returns empty rather than raising when the file has no ``GenParFac`` column:
    a case without AGC data is a case the capacity surrogate covers, not an
    error.  The caller is told which it got.
    """
    emit = log or _noop
    text = _read(path, archive)
    units: Dict[Tuple[int, str], UnitParticipation] = {}
    saw_flag = False

    for header, body in _aux_blocks(text, "Gen"):
        fields = [f.strip().strip('"') for f in header]
        if not any(_GENPARFAC.fullmatch(f) for f in fields):
            continue
        bus_at = _index_of(fields, "BusNum")
        id_at = _index_of(fields, "GenID", "ID")
        fac_at = next(i for i, f in enumerate(fields) if _GENPARFAC.fullmatch(f))
        agc_at = next((i for i, f in enumerate(fields)
                       if _GENAGCABLE.fullmatch(f)), None)
        if bus_at is None or id_at is None:
            continue
        saw_flag = saw_flag or agc_at is not None
        needed = max(bus_at, id_at, fac_at, agc_at if agc_at is not None else 0)
        for row in body:
            tokens = _TOKEN.findall(row)
            if len(tokens) <= needed:
                continue
            try:
                bus = int(tokens[bus_at].strip('"').strip("'"))
                value = float(tokens[fac_at].strip('"').strip("'"))
            except ValueError:
                continue
            uid = tokens[id_at].strip('"').strip("'").strip()
            able = True
            if agc_at is not None:
                able = tokens[agc_at].strip('"').strip("'").strip().upper() == "YES"
            units[(bus, uid)] = UnitParticipation(factor=value, agc_able=able)

    off = sum(1 for u in units.values() if not u.agc_able)
    contradicting = sum(1 for u in units.values()
                        if not u.agc_able and u.factor > 0)
    if not saw_flag:
        emit(f" participation: {len(units)} units carry a GenParFac; no "
             f"GenAGCAble column, so every unit is taken as AGC-able\n")
    else:
        emit(f" participation: {len(units)} units carry a GenParFac, "
             f"{off} are GenAGCAble = NO ({contradicting} of those carry a "
             f"nonzero factor and are held at zero)\n")
    return units


def _index_of(fields: Sequence[str], *names: str) -> Optional[int]:
    lowered = [f.lower() for f in fields]
    for name in names:
        if name.lower() in lowered:
            return lowered.index(name.lower())
    return None


def _aux_blocks(text: str, object_name: str) -> Iterable[Tuple[List[str], List[str]]]:
    """Yield (header fields, data rows) for each ``DATA (<object>, [...])`` block.

    A minimal reader for the one construct this study needs, rather than a
    PowerWorld ``.aux`` parser: the files run to 51 MB and everything else in
    them is unused.
    """
    pattern = re.compile(r"DATA\s*\(\s*" + object_name +
                         r"\b[^)]*?\[(?P<fields>[^\]]*)\]\s*\)\s*\{",
                         re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        fields = [f.strip() for f in match.group("fields").split(",")]
        start = match.end()
        depth, index = 1, start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        rows = [line.strip() for line in text[start:index - 1].splitlines()
                if line.strip() and not line.strip().startswith("//")]
        yield fields, rows


###############################################################################
# Joining onto the network
###############################################################################


def participation_factors(network: Network,
                          aux: Optional[Dict[Tuple[int, str], UnitParticipation]] = None,
                          log: Optional[Callable[[str], None]] = None
                          ) -> Tuple[Dict[int, float], str]:
    """pi_g by generator count, normalized over the in-service AGC-able fleet.

    Returns ``(factors, source)``, where source names which it is -- the AGC
    factors from the ``.aux``, or the capacity surrogate.  The study reports
    the source, because a participation factor invented by the code and one
    read from the distribution are not the same evidence.

    UNIT IDS ARE RESOLVED BY POSITION.  The ``.aux`` is keyed by (bus ID, unit
    ID) and the network by count, and MATPOWER carries no unit ID.  Where a bus
    holds the same number of units in both, they are paired in order -- network
    counts in file order against ``.aux`` identifiers sorted -- which makes the
    join exact and lets ``GenAGCAble`` be applied per unit.  This holds on all
    six ACTIVSg instances.  Where the counts disagree the bus falls back to the
    older behaviour, its AGC-able total split by capacity across its in-service
    units, and the fallback is counted in the log rather than passed over.

    A unit that is not AGC-able gets zero.  It does not respond, so a nonzero
    factor beside it is a contradiction in the file and not a licence to move
    it: on ACTIVSg10k that is 1349 of 2485 units.
    """
    from .postevent import capacity_participation

    emit = log or _noop
    if not aux:
        emit(" participation: no GenParFac available; using the capacity "
             "surrogate\n")
        return capacity_participation(network), "capacity_surrogate"

    # Per bus: the network's unit counts in file order, and the .aux's
    # identifiers sorted.  Sorted on both sides or the pairing is not
    # reproducible across Python builds.
    #
    # ALL units, in service or not.  The .aux lists the out-of-service ones too,
    # so filtering them out here would make the two lists disagree in length at
    # every bus that has one and push it onto the capacity fallback for no
    # reason -- 183 of 1455 buses on ACTIVSg10k before this was fixed.  They are
    # paired, then zeroed below.
    net_at: Dict[int, List[int]] = {}
    for count in sorted(network.gens):
        net_at.setdefault(int(network.gens[count].nodeID), []).append(count)
    aux_at: Dict[int, List[str]] = {}
    for bus_id, uid in aux:
        aux_at.setdefault(int(bus_id), []).append(uid)
    for bus_id in aux_at:
        aux_at[bus_id].sort()

    raw: Dict[int, float] = {count: 0.0 for count in network.gens}
    matched = 0
    exact_buses = 0
    fallback_buses = 0
    zeroed = 0

    for bus_id, counts in net_at.items():
        uids = aux_at.get(bus_id)
        if not uids:
            continue

        if len(uids) == len(counts):
            exact_buses += 1
            for count, uid in zip(counts, uids):
                if not network.gens[count].status:
                    continue                    # out of service: stays at zero
                unit = aux[(bus_id, uid)]
                raw[count] = unit.effective
                matched += 1
                if not unit.agc_able and unit.factor > 0:
                    zeroed += 1
            continue

        # The bus-level fallback.  Only AGC-able factors enter the total, so a
        # bus with no AGC-able unit contributes nothing instead of spreading a
        # factor over units the file says will not move.
        fallback_buses += 1
        live = [c for c in counts if network.gens[c].status]
        if not live:
            continue
        share = sum(aux[(bus_id, uid)].effective for uid in uids)
        capacity = sum(network.gens[c].Pmax for c in live) or 1.0
        for count in live:
            raw[count] = share * (network.gens[count].Pmax / capacity)
            matched += 1

    total = sum(raw.values())
    if matched == 0:
        emit(" participation: the .aux matched no in-service unit; using the "
             "capacity surrogate\n")
        return capacity_participation(network), "capacity_surrogate"
    if total <= 0:
        # Every matched unit is off AGC, or every factor is zero.  Falling back
        # to the capacity surrogate here would make the whole fleet responsive
        # on the strength of a file that says none of it is, which is the one
        # substitution this function must never make.
        raise ValueError(
            f"the .aux matched {matched} in-service units but every effective "
            f"participation factor is zero, so no unit would respond to any "
            f"event. Check GenAGCAble and GenParFac in the case's .aux; this "
            f"is a data problem and the capacity surrogate must not be "
            f"substituted for it.")

    emit(f" participation: matched {matched}/{network.numgens} units "
         f"({exact_buses} buses resolved by unit id, {fallback_buses} by "
         f"capacity split); {zeroed} units held at zero as not AGC-able\n")
    return ({count: value / total for count, value in raw.items()},
            "aux_genparfac_agc")


def unit_inertia(network: Network,
                 dynamics: Dict[Tuple[int, str], UnitDynamics],
                 log: Optional[Callable[[str], None]] = None
                 ) -> Dict[int, UnitDynamics]:
    """Join ``.dyr`` records onto generator counts.

    The MATPOWER file carries no machine ID, so a bus with several units is
    matched by taking that bus's records in order.  A unit with no record gets
    an empty `UnitDynamics`: no inertia and no primary response, which is what
    an unmodelled unit contributes.
    """
    emit = log or _noop
    by_bus: Dict[int, List[UnitDynamics]] = {}
    for (bus, _uid), unit in sorted(dynamics.items()):
        by_bus.setdefault(bus, []).append(unit)

    joined: Dict[int, UnitDynamics] = {}
    taken: Dict[int, int] = {}
    matched = 0
    for count, gen in network.gens.items():
        candidates = by_bus.get(gen.nodeID, [])
        index = taken.get(gen.nodeID, 0)
        if index < len(candidates):
            joined[count] = candidates[index]
            taken[gen.nodeID] = index + 1
            matched += 1
        else:
            joined[count] = UnitDynamics(bus=gen.nodeID, uid="")
    emit(f" inertia: matched {matched}/{network.numgens} units to a machine "
         f"record\n")
    return joined

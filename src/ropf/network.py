"""MATPOWER case files to the network quantities every other module reads.

This module is pinned by ``tests/test_matpower_parity.py``, which compares its
output against MATPOWER 8.1's own ``makeYbus`` and ``makeBdc`` run under Octave
over the same ``.m`` files.  The admittance and DC formulas below are therefore
transcribed rather than derived, and should not be rewritten for style: the
parity is the reason to trust every number the study reports.

One deliberate deviation from MATPOWER is documented at ``Branch.limit``.
"""

from __future__ import annotations

import cmath
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

#: Finite stand-in for an unbounded angle difference.  AMPL needs a bound on
#: ``thetadiff``, so "unconstrained" is expressed as +/- 2*pi rather than as an
#: absent bound.
UNBOUNDED_ANGLE_RAD = 2 * math.pi


class CaseFormatError(ValueError):
    """The case file is not something this reader will silently accept.

    Raised rather than warned so that a malformed or unsupported case stops the
    run instead of producing numbers whose provenance cannot be defended.
    """


def _noop(_message: str) -> None:
    return None


###############################################################################
# Network elements
###############################################################################


class Bus:
    """One row of ``mpc.bus``, with power quantities converted to p.u."""

    def __init__(self, count, nodeID, nodetype, Pd, Qd, Gs, Bs, Vbase, Vmax,
                 Vmin, busline0):
        self.count = count
        self.nodeID = nodeID
        self.nodetype = nodetype
        self.Pd = Pd
        self.Qd = Qd
        self.Gs = Gs
        self.Bs = Bs
        self.Vbase = Vbase
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.busline0 = busline0

        self.genidsbycount: List[int] = []
        self.frombranchids: Dict[int, int] = {}
        self.tobranchids: Dict[int, int] = {}
        self.outdegree = self.indegree = self.degree = 0

    def addgenerator(self, generatorcount: int) -> None:
        self.genidsbycount.append(generatorcount)

    def addfrombranch(self, branchid: int) -> None:
        self.frombranchids[len(self.frombranchids)] = branchid
        self.outdegree += 1
        self.degree += 1

    def addtobranch(self, branchid: int) -> None:
        self.tobranchids[len(self.tobranchids)] = branchid
        self.indegree += 1
        self.degree += 1

    def __repr__(self) -> str:
        return f"Bus(count={self.count}, id={self.nodeID}, type={self.nodetype})"


class Branch:
    """One in-service row of ``mpc.branch``.

    The eight admittance entries reproduce MATPOWER ``makeYbus``; ``bdc``,
    ``Pfinj`` and ``Ptinj`` reproduce ``makeBdc``.  Both are covered by the
    parity test.
    """

    def __init__(self, count, f, id_f, t, id_t, r, x, bc, rateAmva, rateBmva,
                 rateCmva, ratio, angle, maxangle, minangle, status,
                 defaultlimit, branchline0):
        self.count = count
        self.f = f
        self.t = t
        self.id_f = id_f
        self.id_t = id_t
        self.r = r
        self.x = x
        self.bc = bc
        self.status = status
        self.branchline0 = branchline0
        self.rateAmva = rateAmva
        self.rateBmva = rateBmva
        self.rateCmva = rateCmva

        # DELIBERATE DEVIATION FROM MATPOWER.  A rateA of zero means "no rating
        # given".  MATPOWER leaves such a branch unlimited; we substitute a
        # big-M of 2 * sum(Pd) / baseMVA, because AMPL needs a finite bound on
        # Pf and an unbounded variable degrades the DC master's conditioning.
        # The value is non-binding by construction -- twice the total system
        # load cannot flow on one line -- and `constrainedflow` records which
        # branches were substituted so the count can be asserted in tests.
        # This affects 2,462 of 12,706 branches on ACTIVSg10k and 19,590 of
        # 88,207 on ACTIVSg70k, so it is not a rare corner.
        self.limit = rateAmva
        self.constrainedflow = 1
        if self.limit == 0:
            self.limit = defaultlimit
            self.constrainedflow = 0

        if ratio == 0:
            ratio = 1
        self.ratio = ratio
        self.angle = angle
        self.angle_rad = math.pi * angle / 180.0
        self.maxangle = maxangle
        self.minangle = minangle

        # MATPOWER makeAang.  A branch carries angle-difference limits when
        # either side is a real limit, or when exactly one of the two is zero;
        # otherwise the difference is unconstrained.  When it does carry them
        # BOTH bounds apply as given, with a magnitude past 360 read as
        # unbounded.  Freeing a one-sided (-30, 0) limit to 2*pi -- which a
        # naive `if minangle and maxangle` test does -- is wrong: MATPOWER
        # binds that branch at 0.
        constrained = ((minangle != 0 and minangle > -360)
                       or (maxangle != 0 and maxangle < 360)
                       or (minangle != 0 and maxangle == 0)
                       or (minangle == 0 and maxangle != 0))
        if constrained:
            self.loweranglenone = 1 if minangle < -360 else 0
            self.upperanglenone = 1 if maxangle > 360 else 0
            self.minangle_rad = (-UNBOUNDED_ANGLE_RAD if self.loweranglenone
                                 else math.pi * minangle / 180.0)
            self.maxangle_rad = (UNBOUNDED_ANGLE_RAD if self.upperanglenone
                                 else math.pi * maxangle / 180.0)
        else:
            self.loweranglenone = self.upperanglenone = 1
            self.minangle_rad = -UNBOUNDED_ANGLE_RAD
            self.maxangle_rad = UNBOUNDED_ANGLE_RAD

        # --- AC admittance, MATPOWER makeYbus ---------------------------------
        self.invratio2 = invratio2 = 1 / ratio ** 2
        self.multtf = multtf = 1 / (ratio * cmath.exp(1j * self.angle_rad))
        self.multft = multft = 1 / (ratio * cmath.exp(-1j * self.angle_rad))
        self.z = z = r + x * 1j
        self.y = y = 1 / z
        self.Yff = (y + bc / 2 * 1j) * invratio2
        self.Yft = -y * multft
        self.Ytf = -y * multtf
        self.Ytt = y + bc / 2 * 1j
        self.Gff = self.Yff.real
        self.Bff = self.Yff.imag
        self.Gft = self.Yft.real
        self.Bft = self.Yft.imag
        self.Gtf = self.Ytf.real
        self.Btf = self.Ytf.imag
        self.Gtt = self.Ytt.real
        self.Btt = self.Ytt.imag

        # --- the Joule heat coefficient, eq (4c) ------------------------------
        # phi^joule is r_e P_e^2, a dissipation, and a branch does not cool when
        # it is loaded.  A NEGATIVE series resistance is a fitting artefact of a
        # three-winding transformer equivalent and is common in the large
        # synthetic cases: 178 branches on ACTIVSg10k, 447 on ACTIVSg25k and
        # 1,216 on ACTIVSg70k carry one.
        #
        # `r` stays exactly what the case file gives, because the admittance
        # entries above and the MATPOWER parity test both depend on it.
        # `r_heat` is what eq (4c) and cut family (6b) use.  Keeping them
        # separate is not cosmetic: `Phi >= r_e P_e^2` with r_e < 0 is a
        # CONCAVE constraint, so a single negative resistance would turn the
        # master from a convex program into a nonconvex one -- and every rung
        # from ACTIVSg10k upward has one.
        self.r_heat = max(0.0, r)

        # --- DC model, MATPOWER makeBdc ---------------------------------------
        # Computed here, once, rather than at each AMPL setup site.  The two
        # injections are equal and opposite by construction, which is what lets
        # the post-event model carry only the from-end flow (Pt == -Pf holds
        # exactly, phase shifters included).  `test_dc_injections_antisymmetric`
        # asserts it.
        tap = self.ratio if self.ratio != 0.0 else 1.0
        if abs(self.x) > 1e-12:
            self.bdc = (1.0 / self.x) / tap
        else:
            self.bdc = abs(self.Bft)
        self.Pfinj = -self.bdc * self.angle_rad
        self.Ptinj = self.bdc * self.angle_rad

    def __repr__(self) -> str:
        return f"Branch(count={self.count}, {self.f}->{self.t})"


class Generator:
    """One row of ``mpc.gen``, paired with its ``mpc.gencost`` row."""

    def __init__(self, count, nodeID, Pg, Qg, status, Pmax, Pmin, Qmax, Qmin,
                 line0):
        self.count = count
        self.nodeID = nodeID
        self.Pg = Pg
        self.Qg = Qg
        self.status = status
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Qmax = Qmax
        self.Qmin = Qmin
        self.line0 = line0
        self.costlinenum = -1
        #: [quadratic, linear, constant], always length 3.  See `addcost`.
        self.costvector: List[float] = [0.0, 0.0, 0.0]
        self.costdegree = 0

    def addcost(self, costvector: List[float], linenum: int) -> None:
        self.costvector = costvector
        self.costdegree = len(costvector) - 1
        self.costlinenum = linenum

    def __repr__(self) -> str:
        return f"Generator(count={self.count}, bus={self.nodeID})"


###############################################################################
# The parsed case
###############################################################################


@dataclass
class Network:
    """A parsed MATPOWER case.

    Buses, branches and generators are keyed by a 1-based COUNT assigned in file
    order, not by the file's own bus IDs.  Every other module indexes in that
    space -- ``Branch.id_f``/``id_t`` and ``Bus.genidsbycount`` are counts -- so
    the AMPL index sets stay contiguous.  ``id_to_count`` maps the file's IDs in.
    """

    baseMVA: float
    buses: Dict[int, Bus] = field(default_factory=dict)
    branches: Dict[int, Branch] = field(default_factory=dict)
    gens: Dict[int, Generator] = field(default_factory=dict)
    id_to_count: Dict[int, int] = field(default_factory=dict)

    #: Count of the reference bus (the first ``type 3`` row).
    refbus: Optional[int] = None
    #: File ID of the reference bus.
    slackbus: Optional[int] = None

    #: Total demand in MW / MVAr, excluding isolated buses.  ``sumPd`` sets the
    #: big-M substituted for a missing line rating, and the load base the
    #: frequency screen's damping is converted from.
    sumPd: float = 0.0
    sumQd: float = 0.0
    summaxgenP: float = 0.0
    summaxgenQ: float = 0.0

    #: Rows read from ``mpc.branch``, including out-of-service ones, which are
    #: not carried in ``branches``.
    branchcount: int = 0
    numisolated: int = 0
    casefile: Optional[str] = None

    @property
    def numbuses(self) -> int:
        return len(self.buses)

    @property
    def numbranches(self) -> int:
        return len(self.branches)

    @property
    def numgens(self) -> int:
        return len(self.gens)

    @property
    def load_pu(self) -> float:
        """Total demand as p.u. on ``baseMVA``.

        The conversion factor between the textbook load-base damping and the
        system-base damping the swing equation needs.  See
        ``counterfactual.frequency.LoadDamping``.
        """
        return self.sumPd / self.baseMVA

    def unconstrained_branches(self) -> List[int]:
        """Counts of branches whose rating was the big-M substitution."""
        return [c for c, br in self.branches.items() if br.constrainedflow == 0]

    def __repr__(self) -> str:
        return (f"Network({os.path.basename(self.casefile or '?')}: "
                f"{self.numbuses} buses, {self.numgens} gens, "
                f"{self.numbranches} branches)")


###############################################################################
# Parsing
###############################################################################


def read_matpower(casefilename: str,
                  log: Optional[Callable[[str], None]] = None) -> Network:
    """Parse a MATPOWER ``.m`` case file into a `Network`.

    `log` takes a single string; omit it for silence.
    """
    emit = log or _noop
    t0 = time.time()
    emit(f"reading case file {os.path.basename(casefilename)}\n")

    try:
        with open(casefilename, "r") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise CaseFormatError(f"cannot open case file {casefilename}: {exc}") from exc

    net = _parse_lines(lines, emit)
    net.casefile = casefilename
    emit(f"read time: {time.time() - t0:.3f}s\n")
    return net


def _strip_terminator(token: str) -> str:
    """Drop a trailing ``;`` from the last field of a MATPOWER row."""
    return token[:-1] if token.endswith(";") else token


def _parse_lines(lines: List[str], emit: Callable[[str], None]) -> Network:
    net = Network(baseMVA=100.0)
    baseMVA = 100.0
    gencount = 0
    linenum = 2
    numlines = len(lines)

    while linenum <= numlines:
        thisline = lines[linenum - 1].split()
        if not thisline:
            linenum += 1
            continue

        theword = thisline[0]
        if not theword.startswith("mpc."):
            linenum += 1
            continue

        emit(f"found {theword} on line {linenum}\n")

        if theword == "mpc.baseMVA":
            baseMVA = float(_strip_terminator(thisline[2]))
            net.baseMVA = baseMVA
            emit(f" baseMVA: {baseMVA}\n")
            linenum += 1

        elif theword == "mpc.bus":
            linenum = _parse_buses(lines, linenum + 1, net, baseMVA, emit)

        elif theword == "mpc.gen":
            linenum, gencount = _parse_gens(lines, linenum + 1, net, baseMVA, emit)

        elif theword == "mpc.branch":
            linenum = _parse_branches(lines, linenum + 1, net, baseMVA, emit)

        elif theword == "mpc.gencost":
            linenum = _parse_gencost(lines, linenum + 1, net, baseMVA, gencount, emit)

        else:
            linenum += 1

    return net


def _parse_buses(lines, linenum, net: Network, baseMVA, emit) -> int:
    numlines = len(lines)
    numbuses = 0
    slackbus = -1

    while linenum <= numlines:
        thisline = lines[linenum - 1].split()
        if not thisline:
            linenum += 1
            continue
        if thisline[0] == "];":
            emit(f"found end of bus section on line {linenum}\n")
            linenum += 1
            break

        numbuses += 1
        if thisline[1] == "3":
            if slackbus < 0:
                slackbus = int(thisline[0])
                net.refbus = numbuses
                net.slackbus = slackbus
                emit(f" bus {numbuses} ID {thisline[0]} is the reference bus\n")
            else:
                # Several reference buses is legal MATPOWER but ambiguous here,
                # since refbus feeds one theta anchor.  Keep the first.
                emit(f" WARNING: bus {thisline[0]} is a second reference bus;"
                     f" keeping bus ID {slackbus}\n")

        if thisline[0] != "%":
            nodeID, nodetype = int(thisline[0]), int(thisline[1])
            if nodetype not in (1, 2, 3, 4):
                raise CaseFormatError(
                    f"bus {nodeID} has unsupported type {nodetype}")
            if nodetype == 4:
                net.numisolated += 1

            Pd = float(thisline[2])
            Qd = float(thisline[3])
            Gs = float(thisline[4])
            Bs = float(thisline[5])
            Vbase = float(thisline[9])
            Vmax = float(thisline[11])
            Vmin = float(_strip_terminator(thisline[12]))

            # MATPOWER ext2int removes isolated buses outright.  We keep the bus
            # so the AMPL index set stays contiguous, and instead zero what it
            # would contribute: an isolated bus has no incident branch, so a
            # nonzero Pd there makes its own power-balance row infeasible.
            if nodetype == 4 and (Pd or Qd or Gs or Bs):
                emit(f" isolated bus {nodeID} carried Pd {Pd} Qd {Qd} "
                     f"Gs {Gs} Bs {Bs}, zeroed\n")
                Pd = Qd = Gs = Bs = 0.0

            net.buses[numbuses] = Bus(numbuses, nodeID, nodetype, Pd / baseMVA,
                                      Qd / baseMVA, Gs / baseMVA, Bs / baseMVA,
                                      Vbase, Vmax, Vmin, linenum - 1)
            if nodetype in (1, 2, 3):
                net.sumPd += Pd
                net.sumQd += Qd
            net.id_to_count[nodeID] = numbuses

        linenum += 1

    if slackbus < 0:
        emit(" did not find slack bus\n")
    emit(f" {numbuses} buses, sumPd {net.sumPd}, sumQd {net.sumQd}\n")
    if net.numisolated:
        emit(f" isolated: {net.numisolated}\n")
    return linenum


def _parse_gens(lines, linenum, net: Network, baseMVA, emit):
    numlines = len(lines)
    gencount = 0

    while linenum <= numlines:
        thisline = lines[linenum - 1].split()
        if not thisline:
            linenum += 1
            continue
        if thisline[0] == "];":
            emit(f" found end of gen section on line {linenum}\n")
            linenum += 1
            break

        gencount += 1
        # The pglib __api cases terminate the row at field 9.
        if ";" in thisline[9]:
            thisline[9] = thisline[9].split(";")[0]

        nodeID = int(thisline[0])
        Pg = float(thisline[1])
        Qg = float(thisline[2])
        Qmax = float(thisline[3])
        Qmin = float(thisline[4])
        status = 1 if int(thisline[7]) > 0 else 0
        Pmax = float(thisline[8])
        Pmin = float(thisline[9])

        if nodeID not in net.id_to_count:
            raise CaseFormatError(
                f"generator {gencount} sits at nonexistent bus ID {nodeID}")

        idgen = net.id_to_count[nodeID]
        net.gens[gencount] = Generator(gencount, nodeID, Pg, Qg, status,
                                       Pmax / baseMVA, Pmin / baseMVA,
                                       Qmax / baseMVA, Qmin / baseMVA,
                                       linenum - 1)
        net.buses[idgen].addgenerator(gencount)

        # p.u., matching Generator.Pmax/Qmax.  Accumulating these in raw
        # MW/MVAr made every downstream use dimensionally wrong -- it is what
        # produced a +/-23,307 p.u. reactive range at the slack on case118.
        if net.buses[idgen].nodetype in (2, 3):
            net.summaxgenP += Pmax / baseMVA
            net.summaxgenQ += Qmax / baseMVA

        linenum += 1

    busgencount = sum(1 for bus in net.buses.values() if bus.genidsbycount)
    emit(f" {gencount} generators at {busgencount} buses;"
         f" summaxPg {net.summaxgenP} summaxQg {net.summaxgenQ}\n")
    return linenum, gencount


def _parse_branches(lines, linenum, net: Network, baseMVA, emit) -> int:
    numlines = len(lines)
    branchcount = 0
    activebranches = 0
    zerolimit = 0
    # Set from the bus section, which MATPOWER always writes first.
    defaultlimit = 2 * net.sumPd / baseMVA

    while linenum <= numlines:
        thisline = lines[linenum - 1].split()
        if not thisline:
            linenum += 1
            continue
        if thisline[0] == "];":
            emit(f" found end of branch section on line {linenum}\n")
            linenum += 1
            break

        branchcount += 1
        f = int(thisline[0])
        t = int(thisline[1])
        r = float(thisline[2])
        x = float(thisline[3])
        bc = float(thisline[4])
        rateA = float(thisline[5])
        rateB = float(thisline[6])
        rateC = float(thisline[7])
        ratio = float(thisline[8])
        angle = float(thisline[9])
        status = int(thisline[10])
        minangle = float(thisline[11])
        maxangle = float(_strip_terminator(thisline[12]))

        if maxangle < minangle:
            raise CaseFormatError(
                f"branch {branchcount} has maxangle {maxangle} below "
                f"minangle {minangle}")

        if status:
            net.branches[branchcount] = Branch(
                branchcount, f, net.id_to_count[f], t, net.id_to_count[t],
                r, x, bc, rateA / baseMVA, rateB / baseMVA, rateC / baseMVA,
                ratio, angle, maxangle, minangle, status, defaultlimit,
                linenum - 1)
            zerolimit += (net.branches[branchcount].constrainedflow == 0)
            activebranches += 1
            net.buses[net.id_to_count[f]].addfrombranch(branchcount)
            net.buses[net.id_to_count[t]].addtobranch(branchcount)

        linenum += 1

    net.branchcount = branchcount
    emit(f" branchcount: {branchcount} active {activebranches},"
         f" {zerolimit} unconstrained (big-M substituted)\n")
    return linenum


def _parse_gencost(lines, linenum, net: Network, baseMVA, gencount, emit) -> int:
    numlines = len(lines)
    gencostcount = 1

    while linenum <= numlines:
        thisline = lines[linenum - 1].split()
        if not thisline:
            linenum += 1
            continue
        if thisline[0] == "];":
            emit(f" found end of gencost section on line {linenum}\n")
            linenum += 1
            break

        if gencostcount > gencount:
            raise CaseFormatError(
                f"read {gencostcount} gencost rows but only {gencount} generators")

        costtype = int(thisline[0])
        if costtype != 2:
            raise CaseFormatError(
                f"cost of generator {gencostcount} is not polynomial "
                f"(model {costtype}); only model 2 is supported")

        degree = int(thisline[3]) - 1
        if degree > 2 or degree < 0:
            raise CaseFormatError(
                f"degree of cost function for generator {gencostcount} is "
                f"{degree}; only degrees 0 to 2 are supported. A case with an "
                f"n=4 gencost row needs converting to n=3 first.")

        # A MATPOWER gencost row is [2, startup, shutdown, n, c_(n-1) ... c_0],
        # highest power first.  Each term goes into its fixed slot of a length-3
        # [quad, lin, const] vector rather than a degree-length list, so an n=1
        # or n=2 row still loads: consumers index [0], [1] and [2]
        # unconditionally.
        costvector = [0.0, 0.0, 0.0]
        for j in range(degree + 1):
            coeff = float(_strip_terminator(thisline[4 + j]))
            power = degree - j            # this term multiplies Pg**power
            # Pg is p.u. downstream; the file coefficient is per MW.
            costvector[2 - power] = coeff * baseMVA ** power

        net.gens[gencostcount].addcost(costvector, linenum)
        gencostcount += 1
        linenum += 1

    return linenum

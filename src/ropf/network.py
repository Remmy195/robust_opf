"""MATPOWER case files, parsed to match MATPOWER 8.1 makeYbus and makeBdc."""

from __future__ import annotations

import cmath
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

UNBOUNDED_ANGLE_RAD = 2 * math.pi


class CaseFormatError(ValueError):
    pass


class Bus:
    def __init__(self, count, nodeID, nodetype, Pd, Qd, Gs, Bs, Vmax, Vmin):
        self.count = count
        self.nodeID = nodeID
        self.nodetype = nodetype
        self.Pd = Pd
        self.Qd = Qd
        self.Gs = Gs
        self.Bs = Bs
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.genidsbycount: List[int] = []
        self.frombranchids: List[int] = []
        self.tobranchids: List[int] = []


class Branch:
    def __init__(self, count, f, id_f, t, id_t, r, x, bc, rateA, ratio, angle,
                 minangle, maxangle, defaultlimit):
        self.count = count
        self.f = f
        self.t = t
        self.id_f = id_f
        self.id_t = id_t
        self.r = r
        self.x = x

        # rateA = 0 is unlimited in MATPOWER; AMPL needs a finite bound, so a non-binding big-M.
        self.constrainedflow = 1 if rateA != 0 else 0
        self.limit = rateA if rateA != 0 else defaultlimit

        if ratio == 0:
            ratio = 1
        self.ratio = ratio
        self.angle = angle
        self.angle_rad = math.pi * angle / 180.0

        # MATPOWER makeAang.
        constrained = ((minangle != 0 and minangle > -360)
                       or (maxangle != 0 and maxangle < 360)
                       or (minangle != 0 and maxangle == 0)
                       or (minangle == 0 and maxangle != 0))
        if constrained:
            self.minangle_rad = (-UNBOUNDED_ANGLE_RAD if minangle < -360
                                 else math.pi * minangle / 180.0)
            self.maxangle_rad = (UNBOUNDED_ANGLE_RAD if maxangle > 360
                                 else math.pi * maxangle / 180.0)
        else:
            self.minangle_rad = -UNBOUNDED_ANGLE_RAD
            self.maxangle_rad = UNBOUNDED_ANGLE_RAD

        invratio2 = 1 / ratio ** 2
        multtf = 1 / (ratio * cmath.exp(1j * self.angle_rad))
        multft = 1 / (ratio * cmath.exp(-1j * self.angle_rad))
        y = 1 / (r + x * 1j)
        Yff = (y + bc / 2 * 1j) * invratio2
        Yft = -y * multft
        Ytf = -y * multtf
        Ytt = y + bc / 2 * 1j
        self.Gff, self.Bff = Yff.real, Yff.imag
        self.Gft, self.Bft = Yft.real, Yft.imag
        self.Gtf, self.Btf = Ytf.real, Ytf.imag
        self.Gtt, self.Btt = Ytt.real, Ytt.imag

        # Negative r (three-winding fits) would make the Joule cut concave.
        self.r_heat = max(0.0, r)

        self.bdc = (1.0 / x) / ratio if abs(x) > 1e-12 else abs(self.Bft)
        self.Pfinj = -self.bdc * self.angle_rad
        self.Ptinj = self.bdc * self.angle_rad


class Generator:
    def __init__(self, count, nodeID, status, Pmax, Pmin, Qmax, Qmin):
        self.count = count
        self.nodeID = nodeID
        self.status = status
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Qmax = Qmax
        self.Qmin = Qmin
        self.costvector: List[float] = [0.0, 0.0, 0.0]


@dataclass
class Network:
    """Keyed by 1-based count in file order; `id_to_count` maps bus IDs in."""

    baseMVA: float = 100.0
    buses: Dict[int, Bus] = field(default_factory=dict)
    branches: Dict[int, Branch] = field(default_factory=dict)
    gens: Dict[int, Generator] = field(default_factory=dict)
    id_to_count: Dict[int, int] = field(default_factory=dict)
    refbus: Optional[int] = None
    sumPd: float = 0.0
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

    def unconstrained_branches(self) -> List[int]:
        return [c for c, br in self.branches.items() if not br.constrainedflow]

    def __repr__(self) -> str:
        return (f"Network({os.path.basename(self.casefile or '?')}: "
                f"{self.numbuses} buses, {self.numgens} gens, "
                f"{self.numbranches} branches)")


def read_matpower(path: str,
                  log: Optional[Callable[[str], None]] = None) -> Network:
    try:
        with open(path) as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise CaseFormatError(f"cannot open case file {path}: {exc}") from exc

    net = Network(casefile=path)
    sections: Dict[str, List[List[str]]] = {}
    i = 1
    while i < len(lines):
        words = lines[i].split()
        i += 1
        if not words:
            continue
        if words[0] == "mpc.baseMVA":
            net.baseMVA = float(words[2].rstrip(";"))
        elif words[0] in ("mpc.bus", "mpc.gen", "mpc.branch", "mpc.gencost"):
            sections[words[0][4:]], i = _rows(lines, i)

    _buses(net, sections.get("bus", []))
    _gens(net, sections.get("gen", []))
    _branches(net, sections.get("branch", []))
    _gencost(net, sections.get("gencost", []))
    if log:
        log(f"{net}, {len(net.unconstrained_branches())} unrated\n")
    return net


def _rows(lines: List[str], i: int):
    rows = []
    while i < len(lines):
        text = lines[i].split("%", 1)[0].strip()
        i += 1
        if text.startswith("]"):
            break
        if text:
            rows.append(text.replace(";", " ").split())
    return rows, i


def _buses(net: Network, rows) -> None:
    base = net.baseMVA
    for count, row in enumerate(rows, start=1):
        nodeID, nodetype = int(row[0]), int(row[1])
        if nodetype not in (1, 2, 3, 4):
            raise CaseFormatError(f"bus {nodeID} has unsupported type {nodetype}")
        Pd, Qd, Gs, Bs = (float(v) for v in row[2:6])
        if nodetype == 4:
            # MATPOWER drops isolated buses; kept here for contiguous indices.
            net.numisolated += 1
            Pd = Qd = Gs = Bs = 0.0
        else:
            net.sumPd += Pd
        if nodetype == 3 and net.refbus is None:
            net.refbus = count
        net.buses[count] = Bus(count, nodeID, nodetype, Pd / base, Qd / base,
                               Gs / base, Bs / base, float(row[11]),
                               float(row[12]))
        net.id_to_count[nodeID] = count


def _gens(net: Network, rows) -> None:
    base = net.baseMVA
    for count, row in enumerate(rows, start=1):
        nodeID = int(row[0])
        if nodeID not in net.id_to_count:
            raise CaseFormatError(f"generator {count} is at unknown bus {nodeID}")
        net.gens[count] = Generator(count, nodeID, 1 if int(row[7]) > 0 else 0,
                                    float(row[8]) / base, float(row[9]) / base,
                                    float(row[3]) / base, float(row[4]) / base)
        net.buses[net.id_to_count[nodeID]].genidsbycount.append(count)


def _branches(net: Network, rows) -> None:
    base = net.baseMVA
    defaultlimit = 2 * net.sumPd / base
    for count, row in enumerate(rows, start=1):
        f, t = int(row[0]), int(row[1])
        minangle, maxangle = float(row[11]), float(row[12])
        if maxangle < minangle:
            raise CaseFormatError(f"branch {count} has maxangle below minangle")
        if not int(row[10]):
            continue
        id_f, id_t = net.id_to_count[f], net.id_to_count[t]
        net.branches[count] = Branch(
            count, f, id_f, t, id_t, float(row[2]), float(row[3]),
            float(row[4]), float(row[5]) / base, float(row[8]), float(row[9]),
            minangle, maxangle, defaultlimit)
        net.buses[id_f].frombranchids.append(count)
        net.buses[id_t].tobranchids.append(count)
    net.branchcount = len(rows)


def _gencost(net: Network, rows) -> None:
    base = net.baseMVA
    if len(rows) > net.numgens:
        raise CaseFormatError(f"{len(rows)} gencost rows for {net.numgens} generators")
    for count, row in enumerate(rows, start=1):
        if int(row[0]) != 2:
            raise CaseFormatError(f"generator {count}: only polynomial cost (model 2) is supported")
        degree = int(row[3]) - 1
        if not 0 <= degree <= 2:
            raise CaseFormatError(f"generator {count}: cost degree {degree}, only 0 to 2 supported")
        cost = [0.0, 0.0, 0.0]
        for j in range(degree + 1):
            power = degree - j
            cost[2 - power] = float(row[4 + j]) * base ** power
        net.gens[count].costvector = cost

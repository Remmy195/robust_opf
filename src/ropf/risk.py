"""Exposure representations of order q (eq:phibus, eq:phiflow, eq:phijoule) and component selection."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import BusCut
from .network import Network

METRICS = ("max_active_flow", "joule_loss_max", "bus_flow_sum_agg")
FLOW_DOMAINS = ("all", "rated")
SIGN_TOL = 1e-12


def _sign(value: float) -> int:
    if value > SIGN_TOL:
        return 1
    if value < -SIGN_TOL:
        return -1
    return 0


@dataclass
class RiskEval:
    metric: str
    value: float
    components: Dict[int, float] = field(default_factory=dict)
    incidence: Dict[int, List[int]] = field(default_factory=dict)
    branch_sign: Dict[int, int] = field(default_factory=dict)
    q: int = 1

    @property
    def is_bus_family(self) -> bool:
        return self.metric == "bus_flow_sum_agg"

    def ranked(self) -> List[Tuple[int, float]]:
        return sorted(self.components.items(), key=lambda kv: kv[1], reverse=True)


def evaluate(metric: str,
             network: Network,
             Pf: Dict[int, float],
             flow_domain: str = "all",
             q: int = 1) -> RiskEval:
    if metric not in METRICS:
        raise ValueError(f"unknown exposure representation {metric!r}; "
                         f"expected one of {list(METRICS)}")
    if flow_domain not in FLOW_DOMAINS:
        raise ValueError(f"unknown flow domain {flow_domain!r}; "
                         f"expected one of {list(FLOW_DOMAINS)}")
    if q < 1:
        raise ValueError(f"the order q must be at least 1, got {q}")

    if metric == "max_active_flow":
        ev = _evaluate_flow(network, Pf, flow_domain)
    elif metric == "joule_loss_max":
        ev = _evaluate_joule(network, Pf, flow_domain)
    else:
        ev = _evaluate_bus(network, Pf, flow_domain)
    if q > 1:
        ev.q = q
        ev.value = top_q_mean(ev.components.values(), q)
    return ev


def top_q_mean(values: Iterable[float], q: int) -> float:
    """Zero-padded when there are fewer than q values, as t >= 0 pads (cutmean)."""
    return sum(heapq.nlargest(q, values)) / q


def _evaluate_flow(network: Network, Pf: Dict[int, float],
                   flow_domain: str) -> RiskEval:
    components = {count: abs(float(Pf.get(count, 0.0)))
                  for count, branch in network.branches.items()
                  if flow_domain == "all" or branch.constrainedflow}
    return RiskEval("max_active_flow", max(components.values(), default=0.0),
                    components)


def _evaluate_joule(network: Network, Pf: Dict[int, float],
                    flow_domain: str) -> RiskEval:
    components = {}
    for count, branch in network.branches.items():
        if flow_domain != "all" and not branch.constrainedflow:
            continue
        flow = float(Pf.get(count, 0.0))
        components[count] = branch.r_heat * flow * flow
    return RiskEval("joule_loss_max", max(components.values(), default=0.0),
                    components)


def _evaluate_bus(network: Network, Pf: Dict[int, float],
                  flow_domain: str) -> RiskEval:
    """f_i sums |Pf| over incident lines; under "rated" a bus with no rated line drops out."""
    raw_f: Dict[int, float] = {}
    raw_incidence: Dict[int, List[int]] = {}
    branch_sign: Dict[int, int] = {}

    for count, branch in network.branches.items():
        flow = float(Pf.get(count, 0.0))
        branch_sign[count] = _sign(flow)
        if flow_domain != "all" and not branch.constrainedflow:
            continue
        for endpoint in (branch.id_f, branch.id_t):
            raw_f[endpoint] = raw_f.get(endpoint, 0.0) + abs(flow)
            raw_incidence.setdefault(endpoint, []).append(count)

    if flow_domain == "all":
        f_bus = {count: raw_f.get(count, 0.0) for count in network.buses}
        incidence = {count: raw_incidence.get(count, [])
                     for count in network.buses}
    else:
        f_bus = {count: raw_f[count] for count in network.buses
                 if count in raw_f}
        incidence = {count: raw_incidence[count] for count in network.buses
                     if count in raw_f}

    return RiskEval("bus_flow_sum_agg", max(f_bus.values(), default=0.0), f_bus,
                    incidence=incidence, branch_sign=branch_sign)


def bus_cut_key(ev: RiskEval, bus: int) -> Tuple:
    return build_bus_cuts(ev, [int(bus)])[0].key


def select(ev: RiskEval, kappa: int,
           existing: Optional[Iterable] = None) -> List[int]:
    """Algorithm 1 line 5: the kappa largest contributions not already in the pool."""
    if kappa is None or kappa <= 0:
        return []
    seen = set(existing or ())
    chosen: List[int] = []
    for component, _ in ev.ranked():
        key = bus_cut_key(ev, component) if ev.is_bus_family else int(component)
        if key in seen:
            continue
        chosen.append(int(component))
        if len(chosen) >= kappa:
            break
    return chosen


def build_bus_cuts(ev: RiskEval, buses: Sequence[int]) -> List[BusCut]:
    cuts: List[BusCut] = []
    for bus in buses:
        bus = int(bus)
        branch_pos, branch_neg = [], []
        for branch in ev.incidence.get(bus, ()):
            sign = ev.branch_sign.get(branch, 0)
            if sign > 0:
                branch_pos.append(int(branch))
            elif sign < 0:
                branch_neg.append(int(branch))
        cuts.append(BusCut(bus=bus, branch_pos=tuple(branch_pos),
                           branch_neg=tuple(branch_neg)))
    return cuts

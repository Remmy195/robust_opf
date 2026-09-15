"""The risk functionals of Section 2.3 and the separation of Section 2.4.

    phi^bus   = max_i  sum_{(m,n) in E_i} |P_mn|      eq (4a)
    phi^flow  = max_e  |P_e|                          eq (4b)
    phi^joule = max_e  r_e P_e^2                      eq (4c)

f_i is the incident line flows alone: no generation term (Kirchhoff already
puts the injection on those lines) and no demand term (P_di is case data).
Pure functions of plain dicts; nothing here touches AMPL or mutates state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import BusCut
from .network import Network

METRICS = ("max_active_flow", "joule_loss_max", "bus_flow_sum_agg")

#: Component sets a risk functional may be maximized over.  "rated" drops
#: branches with no rateA (ACTIVSg's zero-length bus ties); a bus left with
#: none also drops from the bus functional's max, rather than scoring zero.
FLOW_DOMAINS = ("all", "rated")

#: Below this magnitude a flow contributes no term to a cut.  Zero is a valid
#: subgradient of |.| there and keeps the cut tightest.
SIGN_TOL = 1e-12

#: A component is flagged "bad" when its contribution reaches this fraction of
#: the maximum.  See `exposed`.
EXPOSURE_FRACTION = 0.95


def _sign(value: float, tol: float = SIGN_TOL) -> int:
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


@dataclass
class RiskEval:
    """The active functional at one dispatch.  `value` is rho^k.

    `components` maps each component to its contribution -- branch counts for
    the line functionals, bus counts for the bus functional -- so separation is
    an argmax over it.
    """

    metric: str
    value: float
    components: Dict[int, float] = field(default_factory=dict)
    #: Bus functional only.  `incidence` by bus count, `branch_sign` by branch.
    incidence: Dict[int, List[int]] = field(default_factory=dict)
    branch_sign: Dict[int, int] = field(default_factory=dict)

    @property
    def is_bus_family(self) -> bool:
        return self.metric == "bus_flow_sum_agg"

    @property
    def argmax(self) -> Optional[int]:
        if not self.components:
            return None
        return max(self.components, key=self.components.get)

    def ranked(self) -> List[Tuple[int, float]]:
        return sorted(self.components.items(), key=lambda kv: kv[1], reverse=True)

    def exposed(self, fraction: float = EXPOSURE_FRACTION) -> List[int]:
        return exposed(self, fraction)


def evaluate(metric: str,
             network: Network,
             Pf: Dict[int, float],
             flow_domain: str = "all") -> RiskEval:
    """Algorithm 1 line 2 at k = 0, line 8 thereafter.  `Pf` is by branch count."""
    if metric not in METRICS:
        raise ValueError(f"unknown risk functional {metric!r}; "
                         f"expected one of {list(METRICS)}")
    if flow_domain not in FLOW_DOMAINS:
        raise ValueError(f"unknown flow domain {flow_domain!r}; "
                         f"expected one of {list(FLOW_DOMAINS)}")

    if metric == "max_active_flow":
        return _evaluate_flow(network, Pf, flow_domain)
    if metric == "joule_loss_max":
        return _evaluate_joule(network, Pf, flow_domain)
    return _evaluate_bus(network, Pf, flow_domain)


def _evaluate_flow(network: Network, Pf: Dict[int, float],
                   flow_domain: str = "all") -> RiskEval:
    """eq (4b), maximized over `flow_domain`."""
    components = {count: abs(float(Pf.get(count, 0.0)))
                  for count, branch in network.branches.items()
                  if flow_domain == "all" or branch.constrainedflow}
    return RiskEval("max_active_flow", max(components.values(), default=0.0),
                    components)


def _evaluate_joule(network: Network, Pf: Dict[int, float],
                    flow_domain: str = "all") -> RiskEval:
    """eq (4c), maximized over `flow_domain`.  Uses `Branch.r_heat`, as cut
    family (6b) does; a surrogate and a functional built on different
    coefficients would break eq (11)."""
    components = {}
    for count, branch in network.branches.items():
        if flow_domain != "all" and not branch.constrainedflow:
            continue
        flow = float(Pf.get(count, 0.0))
        components[count] = branch.r_heat * flow * flow
    return RiskEval("joule_loss_max", max(components.values(), default=0.0),
                    components)


def _evaluate_bus(network: Network, Pf: Dict[int, float],
                  flow_domain: str = "all") -> RiskEval:
    """eq (4a) over `flow_domain`.  Every incident line contributes its own
    from-end flow, so both endpoints accumulate |Pf| and neither uses Pt.

    A bus has no rating of its own, so `flow_domain` restricts the incident
    sum each candidate is built from, not the candidate set directly; a bus
    left with no rated incident branch drops out entirely rather than
    summing to zero (matches `study/domain_scan.py`).
    """
    raw_f: Dict[int, float] = {}
    raw_incidence: Dict[int, List[int]] = {}
    branch_sign: Dict[int, int] = {}

    for count, branch in network.branches.items():
        flow = float(Pf.get(count, 0.0))
        branch_sign[count] = _sign(flow)
        if flow_domain != "all" and not branch.constrainedflow:
            continue
        magnitude = abs(flow)
        for endpoint in (branch.id_f, branch.id_t):
            raw_f[endpoint] = raw_f.get(endpoint, 0.0) + magnitude
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


###############################################################################
# Exposure
###############################################################################


def exposed(ev: RiskEval, fraction: float = EXPOSURE_FRACTION) -> List[int]:
    """The components carrying too much exposure: contribution >= fraction*rho.

    Branch counts for the line functionals, bus counts for the bus functional,
    ranked worst first.  `fraction = 1` is the argmax alone.  Empty when the
    functional is zero, since nothing is then exposed.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"the exposure fraction must lie in (0, 1], got "
                         f"{fraction}")
    if ev.value <= 0.0:
        return []
    threshold = fraction * ev.value
    return [component for component, contribution in ev.ranked()
            if contribution >= threshold]


def overloaded(network: Network, Pf: Dict[int, float],
               fraction: float = 1.0) -> List[int]:
    """Rated branches loaded at or above `fraction` of their rating, worst first.

    Independent of the run's functional: a line is overloaded against its own
    rateA whether or not the active functional ranges over it.  Unrated
    branches carry the big-M substitution of `ropf.network` and are skipped.
    """
    loaded = []
    for count, branch in network.branches.items():
        if not branch.constrainedflow or branch.limit <= 0.0:
            continue
        loading = abs(float(Pf.get(count, 0.0))) / branch.limit
        if loading >= fraction:
            loaded.append((count, loading))
    loaded.sort(key=lambda kv: kv[1], reverse=True)
    return [count for count, _ in loaded]


###############################################################################
# Separation
###############################################################################


def bus_cut_key(ev: RiskEval, bus: int) -> Tuple:
    """Exclusion key for the bus family: the bus AND its sign pattern, since a
    repeated bus under a new pattern is a new hyperplane.  Reached by building
    the cut, so the pool and the separation cannot disagree on "the same cut"."""
    return build_bus_cuts(ev, [int(bus)])[0].key


def select(ev: RiskEval, kappa: int,
           existing: Optional[Iterable] = None) -> List[int]:
    """Algorithm 1 line 5: the kappa components of largest contribution.

    `existing` holds what the pool already carries -- branch counts, or
    `bus_cut_key` tuples for the bus family -- and is skipped.
    """
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
    """One eq (6c) cut per bus: kappa separate subgradient inequalities, not one
    dense cut at the argmax."""
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


def cut_value_at(ev: RiskEval, cut: BusCut, Pf: Dict[int, float]) -> float:
    """A bus cut's right-hand side at a dispatch.  At the dispatch it was built
    from this equals f_i exactly, which is what makes eq (6c) valid."""
    total = sum(float(Pf.get(b, 0.0)) for b in cut.branch_pos)
    total -= sum(float(Pf.get(b, 0.0)) for b in cut.branch_neg)
    return total

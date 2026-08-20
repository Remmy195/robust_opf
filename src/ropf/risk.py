"""The risk functionals of Section 2.3, and the separation of Section 2.4.

Three functionals, and nothing else.  Equations (4a) to (4c):

    phi^bus  = max_i  { sum_{(m,n) in E_i} |P_mn| + sum_{g in G_i} |P_g| + |P_di| }
    phi^flow = max_e  |P_e|
    phi^joule= max_e  r_e P_e^2

Everything in this module is a pure function of plain dictionaries.  Nothing
here touches AMPL, logs, or mutates state, which is what lets the same code
evaluate the incumbent inside the loop, score a dispatch in the counterfactual,
and be exercised in tests without a solver.

TWO PROPERTIES OF phi^bus ARE EASY TO GET WRONG AND ARE LOAD-BEARING.

First, ``f_i`` sums *absolute values*, and it includes the generation and demand
at the bus.  Dropping either term, or letting flows cancel, measures something
else entirely: the tell is a degree-2 bus reporting exactly twice the largest
line flow, which is what the branch-only version produced.

Second, the signs enter *only* in the cut, frozen at the incumbent.  That is
what makes eq (6c) a subgradient inequality of ``f_i``, hence a minorant of
phi^bus, hence eq (11).  A cut built from anything other than the incumbent's
sign pattern is not valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import BusCut
from .network import Network

#: The three functionals of Section 2.3.  A metric outside this set is an error
#: rather than a silent fallback: the earlier codebase defaulted an unknown
#: metric to the flow family, which turned a typo in a config into a study that
#: quietly measured the wrong thing.
METRICS = ("max_active_flow", "joule_loss_max", "bus_flow_sum_agg")

#: Below this magnitude a quantity contributes no term to a cut.  It is the
#: subgradient of |.| at zero, where any sign in [-1, 1] is valid; taking zero
#: keeps the cut the tightest valid one.
SIGN_TOL = 1e-12


def _sign(value: float, tol: float = SIGN_TOL) -> int:
    """sigma in eq (6c): the sign the quantity carries at the incumbent."""
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


@dataclass
class RiskEval:
    """The active functional evaluated at one dispatch.

    `value` is rho^k of Algorithm 1.  `components` maps each component to its
    contribution, so the separation is an argmax over it: branch counts for the
    two line functionals, bus counts for the bus functional.
    """

    metric: str
    value: float
    components: Dict[int, float] = field(default_factory=dict)

    # Populated for the bus functional only; all keyed by bus count.
    incidence: Dict[int, List[int]] = field(default_factory=dict)
    branch_sign: Dict[int, int] = field(default_factory=dict)
    gen_incidence: Dict[int, List[int]] = field(default_factory=dict)
    gen_sign: Dict[int, int] = field(default_factory=dict)
    demand: Dict[int, float] = field(default_factory=dict)

    @property
    def is_bus_family(self) -> bool:
        return self.metric == "bus_flow_sum_agg"

    @property
    def argmax(self) -> Optional[int]:
        if not self.components:
            return None
        return max(self.components, key=self.components.get)

    def ranked(self) -> List[Tuple[int, float]]:
        """Components by descending contribution."""
        return sorted(self.components.items(), key=lambda kv: kv[1], reverse=True)


###############################################################################
# Evaluation
###############################################################################


def evaluate(metric: str,
             network: Network,
             Pf: Dict[int, float],
             Pg: Optional[Dict[int, float]] = None) -> RiskEval:
    """Evaluate the active functional at a dispatch.

    This is Algorithm 1 line 2 at k = 0 and line 8 thereafter.  `Pf` is keyed by
    branch count and `Pg` by generator count, matching `ropf.network`.

    `Pg` is required by the bus functional and ignored by the other two.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown risk functional {metric!r}; "
                         f"expected one of {list(METRICS)}")

    if metric == "max_active_flow":
        return _evaluate_flow(network, Pf)
    if metric == "joule_loss_max":
        return _evaluate_joule(network, Pf)
    return _evaluate_bus(network, Pf, Pg or {})


def _evaluate_flow(network: Network, Pf: Dict[int, float]) -> RiskEval:
    """eq (4b): the loading of the most heavily used line."""
    components = {count: abs(float(Pf.get(count, 0.0)))
                  for count in network.branches}
    value = max(components.values(), default=0.0)
    return RiskEval("max_active_flow", value, components)


def _evaluate_joule(network: Network, Pf: Dict[int, float]) -> RiskEval:
    """eq (4c): the ohmic heating of the worst line.

    This is the I^2 R term of the conductor heat balance, so it is the DC active
    flow that enters, not apparent power.
    """
    components = {}
    for count, branch in network.branches.items():
        flow = float(Pf.get(count, 0.0))
        components[count] = branch.r * flow * flow
    value = max(components.values(), default=0.0)
    return RiskEval("joule_loss_max", value, components)


def _evaluate_bus(network: Network,
                  Pf: Dict[int, float],
                  Pg: Dict[int, float]) -> RiskEval:
    """eq (4a): the power incident to the busiest bus.

    Every incident line contributes the flow at *its own from-end*, which is why
    both endpoints of a branch accumulate ``|Pf|`` and neither uses ``Pt``.
    """
    f_bus: Dict[int, float] = {count: 0.0 for count in network.buses}
    incidence: Dict[int, List[int]] = {count: [] for count in network.buses}
    branch_sign: Dict[int, int] = {}
    gen_incidence: Dict[int, List[int]] = {}
    gen_sign: Dict[int, int] = {}
    demand: Dict[int, float] = {}

    for count, branch in network.branches.items():
        flow = float(Pf.get(count, 0.0))
        branch_sign[count] = _sign(flow)
        magnitude = abs(flow)
        for endpoint in (branch.id_f, branch.id_t):
            f_bus[endpoint] += magnitude
            incidence[endpoint].append(count)

    for count, bus in network.buses.items():
        gen_ids = [int(g) for g in bus.genidsbycount]
        gen_incidence[count] = gen_ids
        total = 0.0
        for gen_id in gen_ids:
            output = float(Pg.get(gen_id, 0.0))
            gen_sign[gen_id] = _sign(output)
            total += abs(output)
        pd = abs(float(bus.Pd))
        demand[count] = pd
        f_bus[count] += total + pd

    value = max(f_bus.values(), default=0.0)
    return RiskEval("bus_flow_sum_agg", value, f_bus,
                    incidence=incidence, branch_sign=branch_sign,
                    gen_incidence=gen_incidence, gen_sign=gen_sign,
                    demand=demand)


###############################################################################
# Separation
###############################################################################


def bus_cut_key(ev: RiskEval, bus: int) -> Tuple:
    """Exclusion key for the bus family: the bus AND its sign pattern.

    Section 3 excludes components that already carry a cut, and for the bus
    family does it "by bus and sign pattern together, since a repeated bus under
    a new sign pattern gives a new hyperplane".  Keying on the bus alone refuses
    a genuinely new and valid cut; keying on nothing re-adds the identical
    hyperplane every iteration and the loop stops making progress.

    The key is the cut's own, `BusCut.key`, reached by building the cut this bus
    would give.  The separation therefore skips a component on exactly the
    condition under which the master already holds its hyperplane; two
    independent notions of "the same cut" would eventually disagree.
    """
    return build_bus_cuts(ev, [int(bus)])[0].key


def select(ev: RiskEval,
           kappa: int,
           existing: Optional[Iterable] = None) -> List[int]:
    """Algorithm 1 line 5: the kappa components of largest contribution.

    Returns branch counts for the two line functionals and bus counts for the
    bus functional.  `existing` holds what the pool already carries -- branch
    counts, or `bus_cut_key` tuples for the bus family -- and is skipped over.

    Selection only; the cuts themselves are built by `build_cuts` and appended
    by `ropf.model.Master`.
    """
    if kappa is None or kappa <= 0:
        return []
    seen = set(existing or ())
    chosen: List[int] = []

    for component, contribution in ev.ranked():
        key = bus_cut_key(ev, component) if ev.is_bus_family else int(component)
        if key in seen:
            continue
        chosen.append(int(component))
        if len(chosen) >= kappa:
            break
    return chosen


def build_bus_cuts(ev: RiskEval, buses: Sequence[int]) -> List[BusCut]:
    """Turn selected buses into eq (6c) cuts, one per bus.

    kappa separate cuts, not one dense cut at the argmax.  Each is the
    subgradient inequality of ``f_i`` at the incumbent, so each is valid on its
    own and the pool of them is tighter than any single aggregate.
    """
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

        gen_pos, gen_neg = [], []
        for gen in ev.gen_incidence.get(bus, ()):
            sign = ev.gen_sign.get(gen, 0)
            if sign > 0:
                gen_pos.append(int(gen))
            elif sign < 0:
                gen_neg.append(int(gen))

        cuts.append(BusCut(bus=bus,
                           branch_pos=tuple(branch_pos),
                           branch_neg=tuple(branch_neg),
                           gen_pos=tuple(gen_pos),
                           gen_neg=tuple(gen_neg),
                           demand=float(ev.demand.get(bus, 0.0))))
    return cuts


def cut_value_at(ev: RiskEval, cut: BusCut,
                 Pf: Dict[int, float], Pg: Dict[int, float]) -> float:
    """Evaluate a bus cut's right-hand side at a dispatch.

    At the dispatch the cut was built from this must equal ``f_i`` exactly.
    That identity is the whole justification for eq (6c), so it is checked in
    the tests rather than assumed.
    """
    total = cut.demand
    total += sum(float(Pf.get(b, 0.0)) for b in cut.branch_pos)
    total -= sum(float(Pf.get(b, 0.0)) for b in cut.branch_neg)
    total += sum(float(Pg.get(g, 0.0)) for g in cut.gen_pos)
    total -= sum(float(Pg.get(g, 0.0)) for g in cut.gen_neg)
    return total

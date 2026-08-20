"""The risk functionals of Section 2.3, and the separation of Section 2.4.

Three functionals, and nothing else.  Equations (4a) to (4c):

    phi^bus  = max_i  sum_{(m,n) in E_i} |P_mn|
    phi^flow = max_e  |P_e|
    phi^joule= max_e  r_e P_e^2

Everything in this module is a pure function of plain dictionaries.  Nothing
here touches AMPL, logs, or mutates state, which is what lets the same code
evaluate the incumbent inside the loop, score a dispatch in the counterfactual,
and be exercised in tests without a solver.

``f_i`` IS THE INCIDENT LINE FLOWS AND NOTHING ELSE.  It carries no generation
term and no demand term, and both omissions are deliberate.

GENERATION IS ALREADY IN THE FLOWS.  Kirchhoff at bus i says the injection
there leaves through the incident lines: what a unit produces is precisely what
shows up in ``sum |P_mn|``.  Adding ``sum_g |P_g|`` counts the same power a
second time, and weights a generator bus against a transit bus by an accident
of where the metering happens rather than by how much power moves.

DEMAND IS FIXED DATA.  ``P_di`` is a constant of the case, identical at every
dispatch and at every lambda, so it cannot be traded against anything.  Carried
in ``f_i`` it does not change what any dispatch can do; it only adds a fixed
per-bus offset that reorders the argmax, so the functional would report the
most heavily *loaded* bus rather than the busiest one, and the cut at that bus
would carry a constant the master can never move.

``f_i`` sums *absolute values*, so the flows do not cancel.  A degree-2 bus
carrying P through it scores 2|P|, and that is correct rather than a symptom:
the power crosses two lines.

THE SIGNS ENTER ONLY IN THE CUT, frozen at the incumbent.  That is what makes
eq (6c) a subgradient inequality of ``f_i``, hence a minorant of phi^bus, hence
eq (11).  A cut built from anything other than the incumbent's sign pattern is
not valid.
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

    # Populated for the bus functional only.  `incidence` is keyed by bus
    # count, `branch_sign` by branch count.
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
        """Components by descending contribution."""
        return sorted(self.components.items(), key=lambda kv: kv[1], reverse=True)


###############################################################################
# Evaluation
###############################################################################


def evaluate(metric: str,
             network: Network,
             Pf: Dict[int, float]) -> RiskEval:
    """Evaluate the active functional at a dispatch.

    This is Algorithm 1 line 2 at k = 0 and line 8 thereafter.  `Pf` is keyed by
    branch count, matching `ropf.network`.

    All three functionals are functions of the FLOWS alone.  The bus functional
    took a `Pg` as well while ``f_i`` carried a generation term; it no longer
    does, and the argument is gone rather than ignored, so that a caller cannot
    read the signature as saying the dispatch's generation still matters here.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown risk functional {metric!r}; "
                         f"expected one of {list(METRICS)}")

    if metric == "max_active_flow":
        return _evaluate_flow(network, Pf)
    if metric == "joule_loss_max":
        return _evaluate_joule(network, Pf)
    return _evaluate_bus(network, Pf)


def _evaluate_flow(network: Network, Pf: Dict[int, float]) -> RiskEval:
    """eq (4b): the loading of the most heavily used line."""
    components = {count: abs(float(Pf.get(count, 0.0)))
                  for count in network.branches}
    value = max(components.values(), default=0.0)
    return RiskEval("max_active_flow", value, components)


def _evaluate_joule(network: Network, Pf: Dict[int, float]) -> RiskEval:
    """eq (4c): the ohmic heating of the worst line.

    This is the I^2 R term of the conductor heat balance, so it is the DC active
    flow that enters, not apparent power -- and `Branch.r_heat`, the nonnegative
    heat coefficient, not the raw series resistance, which the large synthetic
    cases give as negative on some transformer equivalents.  The master's cut
    family (6b) uses the same coefficient, and it has to: a surrogate built on
    one coefficient and a functional measured with another would break eq (11).
    """
    components = {}
    for count, branch in network.branches.items():
        flow = float(Pf.get(count, 0.0))
        components[count] = branch.r_heat * flow * flow
    value = max(components.values(), default=0.0)
    return RiskEval("joule_loss_max", value, components)


def _evaluate_bus(network: Network, Pf: Dict[int, float]) -> RiskEval:
    """eq (4a): the power crossing the busiest bus.

    Every incident line contributes the flow at *its own from-end*, which is why
    both endpoints of a branch accumulate ``|Pf|`` and neither uses ``Pt``.  See
    the module docstring for why generation and demand are not terms here.
    """
    f_bus: Dict[int, float] = {count: 0.0 for count in network.buses}
    incidence: Dict[int, List[int]] = {count: [] for count in network.buses}
    branch_sign: Dict[int, int] = {}

    for count, branch in network.branches.items():
        flow = float(Pf.get(count, 0.0))
        branch_sign[count] = _sign(flow)
        magnitude = abs(flow)
        for endpoint in (branch.id_f, branch.id_t):
            f_bus[endpoint] += magnitude
            incidence[endpoint].append(count)

    value = max(f_bus.values(), default=0.0)
    return RiskEval("bus_flow_sum_agg", value, f_bus,
                    incidence=incidence, branch_sign=branch_sign)


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

        cuts.append(BusCut(bus=bus,
                           branch_pos=tuple(branch_pos),
                           branch_neg=tuple(branch_neg)))
    return cuts


def cut_value_at(ev: RiskEval, cut: BusCut,
                 Pf: Dict[int, float]) -> float:
    """Evaluate a bus cut's right-hand side at a dispatch.

    At the dispatch the cut was built from this must equal ``f_i`` exactly.
    That identity is the whole justification for eq (6c), so it is checked in
    the tests rather than assumed.
    """
    total = sum(float(Pf.get(b, 0.0)) for b in cut.branch_pos)
    total -= sum(float(Pf.get(b, 0.0)) for b in cut.branch_neg)
    return total

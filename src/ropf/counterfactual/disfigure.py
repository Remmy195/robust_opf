"""The two disfigurement classes of Section 4.1.

    top_k_units    the K generating units of highest output at the NOMINAL
                   dispatch.  The ranking is taken once, at lambda = 0, and
                   held fixed across every dispatch.
    random_walk    a connected set of K network components, buses and lines
                   together, collected by a susceptance-weighted walk.

WHY THE RANKING IS FIXED.  Section 4.1.1 ranks at lambda = 0 and holds that
ranking across every dispatch, which is what makes the comparison paired: the
same units are removed from each, so the difference that remains is
attributable to lambda.  Re-ranking per dispatch would remove different units
from different dispatches and measure something else -- and it would flatter the
de-risked ones, since a de-risked dispatch has already spread its output and its
top-K carries less.  `top_k_units` therefore takes the ranking dispatch as a
separate argument from the dispatch under test, so a caller cannot pass one by
accident.

WHAT COUNTS TOWARD K IN THE WALK.  Buses and lines together, per Section 4.1.2.
A unit removed with its bus is not a component in its own right.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..network import Network
from .postevent import Disfigurement


###############################################################################
# Class (a): loss of the largest units
###############################################################################


def rank_units(P_star: Dict[int, float]) -> List[int]:
    """Generator counts by descending output at the ranking dispatch.

    Ties break on the count, so the ranking is deterministic: a tie broken by
    dictionary order would make the campaign irreproducible on a different
    Python build.
    """
    return [count for count, _ in
            sorted(P_star.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))]


def top_k_units(ranking: Sequence[int], K: int) -> Disfigurement:
    """Section 4.1.1: disable the K highest-output units of `ranking`.

    `ranking` comes from `rank_units` on the NOMINAL dispatch and is the same
    list for every dispatch the campaign evaluates.  The network stays connected
    under this class, so the whole loss falls on the frequency screen.
    """
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")
    return Disfigurement(gens=frozenset(int(g) for g in ranking[:K]),
                         label=f"top{K}")


###############################################################################
# Class (b): random-walk component sets
###############################################################################


def random_walk(network: Network,
                K: int,
                rng: random.Random,
                max_steps: Optional[int] = None) -> Disfigurement:
    """Section 4.1.2: a connected set of K components, buses and lines together.

    The walk seeds at a uniformly drawn bus and steps from bus i across the
    incident line (i,j) with probability |B_ij| / sum_n |B_in|, collecting each
    line it crosses and each bus it reaches until K distinct components are
    held.  Susceptance weighting makes the walk favour electrically short steps,
    which is the property the commute-time argument rests on.

    The line crossed is collected before the bus reached, so the set never
    overshoots K.  A walk that runs out of moves -- a dead-end bus in a
    radially attached pocket -- reseeds rather than returning short, since a set
    of fewer than K components is not the object Section 4.1.2 defines.
    """
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")

    incident = _incidence(network)
    seedable = [bus for bus, edges in incident.items() if edges]
    if not seedable:
        raise ValueError("the network has no branch to walk along")

    limit = max_steps if max_steps is not None else 100 * K
    current = rng.choice(seedable)
    buses = {current}
    lines: set = set()
    steps = 0

    while len(buses) + len(lines) < K and steps < limit:
        steps += 1
        options = incident.get(current) or []
        if not options:
            current = rng.choice(seedable)
            buses.add(current)
            continue

        line, nxt = _weighted_step(options, rng)
        if len(buses) + len(lines) < K:
            lines.add(line)
        if len(buses) + len(lines) < K:
            buses.add(nxt)
        current = nxt

    return Disfigurement(buses=frozenset(buses), branches=frozenset(lines),
                         label=f"walk{K}")


def _incidence(network: Network) -> Dict[int, List[Tuple[int, int, float]]]:
    """Per bus, the incident lines as (branch count, other bus, |B|).

    A branch of zero susceptance is dropped: it can never be stepped across, and
    leaving it in would put a zero in the weight vector.
    """
    incident: Dict[int, List[Tuple[int, int, float]]] = {
        bus: [] for bus in network.buses}
    for count, branch in network.branches.items():
        weight = abs(branch.bdc)
        if weight <= 0.0:
            continue
        f, t = int(branch.id_f), int(branch.id_t)
        incident[f].append((int(count), t, weight))
        incident[t].append((int(count), f, weight))
    return incident


def _weighted_step(options: Sequence[Tuple[int, int, float]],
                   rng: random.Random) -> Tuple[int, int]:
    """Draw one incident line with probability proportional to |B|."""
    total = sum(weight for _, _, weight in options)
    draw = rng.random() * total
    accumulated = 0.0
    for line, nxt, weight in options:
        accumulated += weight
        if draw <= accumulated:
            return line, nxt
    line, nxt, _ = options[-1]                 # floating point ran us past the end
    return line, nxt


def walk_draws(network: Network,
               K: int,
               draws: int,
               seed: int) -> List[Disfigurement]:
    """`draws` independent walks at size K, from one seed.

    The seed is explicit and the generator is local, so a campaign is
    reproducible and two campaigns at different K do not consume each other's
    random stream.
    """
    rng = random.Random(seed)
    return [random_walk(network, K, rng) for _ in range(draws)]

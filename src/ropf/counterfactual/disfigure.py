"""The two sampled disfigurement classes of Section 4.1.

    top_k_units    the K units of highest output at the NOMINAL dispatch
    random_walk    a connected set of K components, buses and lines together,
                   collected by a susceptance-weighted walk

WHY THE RANKING IS FIXED.  Ranking once at lambda = 0 is what makes the
comparison paired: the same units are removed from every dispatch, so what
remains is attributable to lambda.  Re-ranking would flatter the de-risked
dispatches, whose top-K carries less by construction, so `top_k_units` takes the
ranking dispatch as a separate argument from the dispatch under test.

Buses and lines both count toward K; a unit removed with its bus does not.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..network import Network
from .postevent import Disfigurement


###############################################################################
# Class (a): loss of the largest units
###############################################################################


def rank_units(P_star: Dict[int, float]) -> List[int]:
    """Generator counts by descending output.  Ties break on the count, so the
    ranking does not depend on dictionary order."""
    return [count for count, _ in
            sorted(P_star.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))]


def top_k_units(ranking: Sequence[int], K: int) -> Disfigurement:
    """Section 4.1.1: disable the K highest-output units of `ranking`, which comes
    from `rank_units` on the NOMINAL dispatch and is the same list for every
    dispatch.  The network stays connected, so the loss falls on the screen."""
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

    Seeds at a uniform bus and steps across the incident line (i,j) with
    probability |B_ij| / sum_n |B_in|, collecting each line crossed and each bus
    reached.  Susceptance weighting favours electrically short steps, which is
    what the commute-time argument rests on.

    The line is collected before the bus, so the set never overshoots K.  A walk
    out of moves reseeds rather than returning short: fewer than K components is
    not the object Section 4.1.2 defines.
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
    """Per bus, the incident lines as (branch count, other bus, |B|).  A branch of
    zero susceptance is dropped: it can never be stepped across."""
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
    """`draws` independent walks at size K, from one seed.  The generator is
    local, so two campaigns at different K do not share a stream."""
    rng = random.Random(seed)
    return [random_walk(network, K, rng) for _ in range(draws)]

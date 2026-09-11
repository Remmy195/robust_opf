"""Frozen-injection post-event DC flows: (D) in closed form at gamma = 0.

At ``gamma = 0`` the eq (6d) window collapses to a point, pinning every survivor
at ``clamp(P*_g, Pmin_g, Pmax_g)``.  For an event removing no injection --- a
branch outage, or a contingency naming an already-offline unit --- nothing else
may move either: `Pbalance` summed over every bus annihilates the flow terms and
leaves ``sum(L) = sum(Pd) + sum(Gs) - sum(Pg)``, the master own balance residual
at pinned ``Pg``.  The flows are DETERMINED by ``P*`` and the surviving
topology: one sparse linear solve, no LP.

THIS IS NOT A SCREEN AND NOT A BOUND.  The low-rank update of
COUNTERFACTUAL_STANDARD 3.7 is a sufficient condition falling through to the LP,
which is right for ``gamma > 0`` and wrong here: with no dispatch decision left
this is the exact answer.  `tests/test_counterfactual.py` asserts the two paths
agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..network import Network


def flows(net: Network,
          buses: Sequence[int],
          index: Dict[int, int],
          live_branches: Iterable[int],
          injection: Dict[int, float],
          ref: Optional[int] = None) -> Dict[int, float]:
    """Post-event DC flows with the injections frozen.  Returns {branch: Pf}.

    Sparse, because the top instance solves this ~20,000 times and a dense
    factorization of a 2000-bus B is most of a second each.  The reference
    angle is fixed by deleting its row and column, which is what `ref_angle`
    does in `postevent.mod`: the network is connected here, so that choice
    fixes the gauge and nothing else.

    ``ref`` is the bus count to gauge at, and defaults to the first of
    ``buses``.  It is an argument rather than a constant because the LP fixes
    ``theta`` at `PostEvent._reference`'s choice, and in exact arithmetic the
    two gauges give the same flows only when the injections sum to zero.  They
    sum to the master's balance residual, so the two paths agree to the residual
    if they gauge at the same bus and to rather less if they do not.
    """
    n = len(buses)
    rhs = np.array([injection.get(b, 0.0) for b in buses])
    rows, cols, vals = [], [], []

    live = sorted(live_branches)
    for e in live:
        br = net.branches[e]
        f, t = index[br.id_f], index[br.id_t]
        rows += [f, t, f, t]
        cols += [f, t, t, f]
        vals += [br.bdc, br.bdc, -br.bdc, -br.bdc]
        rhs[f] -= br.Pfinj
        rhs[t] += br.Pfinj

    B = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    gauge = index[ref] if ref is not None else 0
    keep = np.array([i for i in range(n) if i != gauge])
    theta = np.zeros(n)
    theta[keep] = spla.spsolve(B[keep][:, keep].tocsc(), rhs[keep])

    return {e: net.branches[e].bdc * (theta[index[net.branches[e].id_f]]
                                      - theta[index[net.branches[e].id_t]])
               + net.branches[e].Pfinj
            for e in live}


###############################################################################
# The severity numbers, computed the same way on both paths
###############################################################################


@dataclass(frozen=True)
class Loading:
    """What a set of post-event flows does to the ratings.

    ``worst_loading`` is the parameter-free one and is the number to lead with:
    it does not mention ``beta``, so a campaign run once can be re-read at any
    emergency rating factor without re-solving.  The other two do mention it and
    must never be quoted without it.
    """

    #: max_e |Pf_e| / U_e over RATED surviving branches, dimensionless.
    worst_loading: float
    #: The branch attaining it, or None when no surviving branch is rated.
    worst_branch: Optional[int]
    #: max_e s_e, p.u.  GO3's largest s_jtk^+.
    overload_max_pu: float
    #: sum_e s_e, p.u.  GO3's summed violation penalty.
    overload_sum_pu: float
    #: Surviving rated branches past beta * U.
    n_over: int


#: Relative tolerance on one branch's rating before a flow past it is called a
#: violation.  Same form and same constant as `results.BOUND_RTOL`, and for the
#: same reason: looser than the LP solver's own feasibility tolerance, tighter
#: than anything physical.  A cost-phase solution respects ``|Pf| <= beta U`` to
#: the solver's tolerance and no better, so without this every such solve would
#: report a violation of a few times 1e-7 p.u. -- a few hundredths of a watt.
OVERLOAD_RTOL = 1e-6


def summarize(net: Network, pf: Dict[int, float], beta: float) -> Loading:
    """The three severity numbers of a post-event flow pattern.

    THE VIOLATION IS READ OFF THE FLOWS, NOT OFF `s`.  ``s_e`` is
    ``max(0, |Pf_e| - beta U_e)``, which is what the AMPL variable equals at the
    overload phase's optimum and is what the returned dispatch actually carries
    in every other phase, where ``s`` is only bounded by the budget row and may
    sit above its own minimum.  Computing it here makes one definition serve
    both paths.

    UNRATED BRANCHES ARE SKIPPED.  `ropf.network` substitutes a big-M of
    ``2 sum(Pd) / baseMVA`` for a branch whose rateA is zero -- 19,590 of 88,207
    on ACTIVSg70k, so not a rare corner.  That number is a bound AMPL needs, not
    a rating, and dividing a flow by it would report a loading against a
    quantity the case never stated.  It is non-binding by construction, so
    excluding those branches removes no violation that was ever real.
    """
    worst, worst_branch = 0.0, None
    over_max, over_sum, n_over = 0.0, 0.0, 0
    for count, value in pf.items():
        branch = net.branches[count]
        if branch.constrainedflow == 0:
            continue
        rating = branch.limit
        loading = abs(value) / rating
        if loading > worst:
            worst, worst_branch = loading, count
        excess = abs(value) - beta * rating
        if excess > OVERLOAD_RTOL * max(1.0, beta * rating):
            over_max = max(over_max, excess)
            over_sum += excess
            n_over += 1
    return Loading(worst_loading=worst, worst_branch=worst_branch,
                   overload_max_pu=over_max, overload_sum_pu=over_sum,
                   n_over=n_over)

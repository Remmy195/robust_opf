###############################################################################
#
#  (D) -- the post-event model.
#
#  Section 4.2 of the manuscript, equations (6a)-(6g) of that section.  Given a
#  pre-event dispatch P* and a disfigurement, decide whether the surviving
#  network can serve its demand, and report the lost load if it cannot.
#
#  THE DISFIGUREMENT ENTERS AS DATA, NOT AS SET MEMBERSHIP.  The full network
#  stays in the model and three binary parameters mark what survives.  This is
#  deliberate: a counterfactual campaign solves this model tens of thousands of
#  times, and changing set membership would force AMPL to rebuild the model on
#  every one of them.  Nothing here is ever deleted or redeclared between
#  solves; the driver updates parameters and re-solves.
#
#  Two exclusions happen in Python BEFORE this model is reached, and neither is
#  representable here: a disfigurement that disconnects the network leaves the
#  sample entirely (Section 4.2), and one that fails the frequency screen is a
#  collapse.  See ropf/counterfactual/.
#
###############################################################################

# --- sets, the FULL pre-event network ----------------------------------------
set buses;
set gens;
set branches;
set branches_f {i in buses};
set branches_t {i in buses};
set bus_gens   {i in buses};

# --- network data ------------------------------------------------------------
param bus_f {e in branches};
param bus_t {e in branches};
param bdc   {e in branches};
param Pfinj {e in branches} default 0;
param U     {e in branches} >= 0;
param Pd    {i in buses};
param Gs    {i in buses} default 0;

param c2 {g in gens};      # quadratic cost, $/p.u.^2
param c1 {g in gens};      # linear cost,    $/p.u.
param c0 {g in gens};      # no-load cost,   $

# --- the disfigurement -------------------------------------------------------
# A removed bus deletes every line incident to it and every unit located at it;
# a removed line deletes that line alone.  The driver expands the removal set
# into these three vectors.
param alive_bus {i in buses} binary default 1;
param alive_br  {e in branches} binary default 1;
param alive_gen {g in gens} binary default 1;

# --- declared study parameters -----------------------------------------------
# beta, eq (6e): the emergency rating factor.  1 is the NORMAL rating.
#
# THE PER-BRANCH MAGNITUDE IS MISSING FROM THE DATA; THE MONITORING RULE IS NOT.
# rateB and rateC are zero across every ACTIVSg case in .m, .aux and .RAW, so
# there is no emergency rating to read off a branch.  But every one of the six
# .aux files carries the same LimitSet record -- rate set "A" for the base case
# and "A" again for the CONTINGENCY case, at LSLinePercent 100 -- so the case
# authors' own contingency rule is rating A at 100 percent, which is beta = 1.
# See COUNTERFACTUAL_STANDARD.md section 5.3 for the record and section 3.5 for
# what follows from it: 1 is the default and is cited to the case file, and 1.2
# is a declared short-term-overload sensitivity swept beside it.
#
# The default below is 1 for the same reason it always was: a model file that
# silently applied a 20% overload would make every unset caller report less
# violation than the case it was given supports.  It is now also the study's
# primary point rather than a placeholder.
param beta >= 1 default 1;

# Phase switch, eq (6f).  0 holds L at zero for the cost and overload phases;
# 1 frees it for the load-shed phase.  Driving this by a parameter rather than
# by fixing variables keeps every phase in one generated problem.
param shed_allowed binary default 0;

# The lexicographic budget on the overload, eq (6e').  The phases run in a
# fixed priority order -- overload first, then cost or lost load at that
# overload -- and this row is how a later phase is held at the earlier one's
# optimum without fixing a variable or regenerating the model.
#
#   s_capped = 0                the row reads 0 <= s_cap and is vacuous; the
#                               overload phase minimizes sum(s) unconstrained
#   s_capped = 1, s_cap = 0     s is driven to zero, which is the HARD rating
#                               of eq (6e) and is the cost phase
#   s_capped = 1, s_cap = s*    the load-shed phase, held at the minimum
#                               overload the overload phase found
#
# Same idiom as `dead_angle` and for the same reason: the row is always
# present and is made vacuous by a parameter, so nothing here ever forces AMPL
# to regenerate the constraint set between solves.
param s_cap >= 0 default 0;
param s_capped binary default 1;

# eq (6d): the response window, already intersected with the unit's operating
# range.  The intersection max(Pmin, P* - pi_g*gamma), min(Pmax, P* + pi_g*gamma)
# and the guard for a P* lying a hair outside [Pmin, Pmax] are computed in
# Python, where they are covered by tests, rather than written as an AMPL
# expression that cannot be exercised on its own.  Note (D) is never asked to
# resolve a generation SURPLUS -- see the Python-side evaluate() docstring --
# so this window only ever needs to bound a genuine deficit response.
param Pg_lo {g in gens};
param Pg_hi {g in gens};

# eq (6g): a surviving reference bus.  The network is connected by the time this
# model is reached, so the choice only fixes the gauge.  Not `symbolic`: bus
# index sets in this study are numeric counts, and a symbolic parameter over a
# numeric set is accepted but means something else.
param ref_bus in buses;

# --- variables ---------------------------------------------------------------
# Pf CARRIES NO RATING BOUND.  It used to, and that was the bug: at gamma = 0
# the response window pins Pg at P*, so for a branch outage the post-event
# flows are DETERMINED and a bound on them does not measure the overload, it
# makes the LP infeasible.  Phase 2 could not rescue it either -- summing
# `Pbalance` over every bus annihilates every Pf term and leaves
# sum(L) = sum(Pd) + sum(Gs) - sum(Pg), which at pinned Pg is the master's own
# balance residual, so L is pinned at zero and the load-shed phase's feasible
# set IS the cost phase's.  The contingencies that mattered most returned no
# number at all.  The rating is now soft: `s` is GO3's s_jtk^+, the violation
# is priced by the `overload` objective rather than forbidden, and every
# admitted draw returns a magnitude.  GO3 (157)-(160) does it the same way.
var theta {i in buses};
var Pf {e in branches};
var s   {e in branches} >= 0;      # GO3 s_jtk^+, the rating violation, p.u.
var Pg {g in gens} >= alive_gen[g] * Pg_lo[g],
                   <= alive_gen[g] * Pg_hi[g];
var L  {i in buses} >= 0, <= shed_allowed * alive_bus[i] * Pd[i];

# --- objectives ---------------------------------------------------------------
# THREE OBJECTIVES, RUN LEXICOGRAPHICALLY, THE RATING VIOLATION FIRST.  That
# order is the one the hard-rating model already had -- it would shed any
# amount of load to respect a rating -- written out so that the case where no
# amount of shedding can respect one returns a number instead of `infeasible`.
#
#   overload   min sum(s).  The smallest rating violation the surviving
#              network admits.  Zero on almost every draw.
#   cost       min gen_cost with s held at zero, which is eq (6e) as a hard
#              bound.  Reached whenever a zero-overload, zero-shed operating
#              point exists, and it is then the ONLY solve the evaluation
#              needs -- the driver tries it first and falls back.
#   lost_load  min sum(L) with sum(s) held at the overload phase's optimum.
#              Reached when the cost phase has no solution.
#
# The load-shed phase is asked ONLY when L can move.  For a branch-only event
# at gamma = 0 it cannot: `Pbalance` summed over every bus pins sum(L) at the
# master's balance residual, so its feasible set is the cost phase's and asking
# it proves nothing.  The driver gates on that rather than letting the solve run
# and reporting its failure as a contingency.  See `PostEvent.evaluate`.
minimize overload:
    sum {e in branches} s[e];

minimize gen_cost:
    sum {g in gens} alive_gen[g] * (c0[g] + c1[g]*Pg[g] + c2[g]*Pg[g]^2);

minimize lost_load:
    sum {i in buses} L[i];

# --- eq (6g) ------------------------------------------------------------------
subject to ref_angle:
    theta[ref_bus] = 0;

# A removed bus carries no flow, no unit and no load, leaving its angle a free
# direction in the LP.  Pinned algebraically rather than by conditional
# indexing, so that changing alive_bus does not force AMPL to regenerate the
# constraint set: the row is vacuous (0 = 0) wherever the bus survives.
subject to dead_angle {i in buses}:
    (1 - alive_bus[i]) * theta[i] = 0;

# --- eq (6b): DC flow definition ----------------------------------------------
# alive_br multiplies the whole right side, so a removed line carries zero flow
# while the angle difference across it stays free.
subject to Pf_def {e in branches}:
    Pf[e] = alive_br[e] * (bdc[e]*(theta[bus_f[e]] - theta[bus_t[e]]) + Pfinj[e]);

# --- eq (6e), soft: the rating is priced, not forbidden ------------------------
# A dead branch keeps alive_br = 0, so `Pf_def` already forces Pf = 0 there and
# the two rows read 0 <= s and -s <= 0: s goes to zero with the branch and no
# conditional indexing is needed.  An unrated branch carries the big-M of
# `ropf.network` in U, which is non-binding by construction, so s is zero there
# too and the big-M never enters a reported violation.
subject to rating_hi {e in branches}:
    Pf[e] <=  beta * alive_br[e] * U[e] + s[e];

subject to rating_lo {e in branches}:
    Pf[e] >= -beta * alive_br[e] * U[e] - s[e];

# --- eq (6e'): the lexicographic budget ---------------------------------------
# Vacuous at s_capped = 0.  See the parameter block above for the three states.
subject to overload_budget:
    s_capped * (sum {e in branches} s[e]) <= s_cap;

# --- eq (6c): balance, with the shed entering as a reduction in demand --------
# From-end flows at both ends, as in (M): the DC model is lossless and the phase
# shift injections cancel, so the to-end flow is exactly -Pf.
#
# alive_bus multiplies the demand and the shunt because a removed bus takes its
# load with it.  Without that factor the row at a removed bus reads
# 0 = -Pd_i - Gs_i, which is infeasible at every removed bus carrying load --
# so a disfigurement that removed any load bus would be reported as a system
# with no feasible post-event dispatch rather than as what it is.  L_i is
# already held at zero there by its own bound, so the demand a removed bus
# takes with it is NOT counted as lost load; eq (6f) sums L over the SURVIVING
# buses only.  The row stays present and vacuous, so changing alive_bus never
# makes AMPL regenerate the constraint set.
subject to Pbalance {i in buses}:
    (sum {e in branches_f[i]} Pf[e]) - (sum {e in branches_t[i]} Pf[e])
    = (sum {g in bus_gens[i]} Pg[g])
      - alive_bus[i] * (Pd[i] - L[i]) - alive_bus[i] * Gs[i];

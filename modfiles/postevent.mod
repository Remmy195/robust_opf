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
# beta, eq (6e): the emergency rating factor.  Defaults to 1, the NORMAL rating.
# No case in this study carries a usable emergency rating -- rateB and rateC are
# zero across every ACTIVSg case in .m, .aux and .RAW, and the ACTIVSg2000
# LimitSet monitors contingencies against rating "A" -- so a value above 1
# requires an external, cited source.
param beta >= 1 default 1;

# Phase switch, eq (6f).  0 holds L at zero for the cost phase; 1 frees it for
# the load-shed phase.  Driving this by a parameter rather than by fixing
# variables keeps both phases in one generated problem.
param shed_allowed binary default 0;

# eq (6d): the response window, already intersected with the unit's operating
# range.  The intersection max(Pmin, P* - pi_g*gamma), min(Pmax, P* + pi_g*gamma)
# and the guard for a P* lying a hair outside [Pmin, Pmax] are computed in
# Python, where they are covered by tests, rather than written as an AMPL
# expression that cannot be exercised on its own.
param Pg_lo {g in gens};
param Pg_hi {g in gens};

# eq (6g): a surviving reference bus.  The network is connected by the time this
# model is reached, so the choice only fixes the gauge.
param ref_bus symbolic in buses;

# --- variables ---------------------------------------------------------------
var theta {i in buses};
var Pf {e in branches} >= -beta * alive_br[e] * U[e],
                       <=  beta * alive_br[e] * U[e];
var Pg {g in gens} >= alive_gen[g] * Pg_lo[g],
                   <= alive_gen[g] * Pg_hi[g];
var L  {i in buses} >= 0, <= shed_allowed * alive_bus[i] * Pd[i];

# --- objectives ---------------------------------------------------------------
# Phase 1 minimizes cost with L pinned at zero.  If that is infeasible there is
# no zero-shed operating point inside the response window and the ratings, and
# phase 2 minimizes lost load instead -- which always has a solution, since
# shedding every bus is feasible.  An infeasible phase 2 is a modelling error,
# not a hard contingency.
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

# --- eq (6c): balance, with the shed entering as a reduction in demand --------
# From-end flows at both ends, as in (M): the DC model is lossless and the phase
# shift injections cancel, so the to-end flow is exactly -Pf.
subject to Pbalance {i in buses}:
    (sum {e in branches_f[i]} Pf[e]) - (sum {e in branches_t[i]} Pf[e])
    = (sum {g in bus_gens[i]} Pg[g]) - (Pd[i] - L[i]) - Gs[i];

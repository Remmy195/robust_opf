###############################################################################
#
#  (M) -- the DC master problem.
#
#  Section 2.1 of the manuscript, equations (1a)-(1e), plus the risk cuts of
#  Section 2.4, equations (6a)-(6c).
#
#  Naming follows the paper, not the solver: `risk_weight` is lambda, the price
#  on the risk surrogate, and `Phi` is the surrogate itself.  `theta` is a
#  voltage angle and nothing else.
#
###############################################################################

# --- sets --------------------------------------------------------------------
set buses;
set gens;
set branches;
set branches_f {i in buses};      # branches whose FROM end is bus i
set branches_t {i in buses};      # branches whose TO end is bus i
set bus_gens   {i in buses};

# --- network data ------------------------------------------------------------
param bus_f {e in branches};
param bus_t {e in branches};
param bdc   {e in branches};      # DC susceptance, (1/x)/tap
param Pfinj {e in branches} default 0;   # phase-shift injection, -bdc * shift
param U     {e in branches} >= 0;        # line rating, p.u.
# The Joule HEAT coefficient of eq (4c): max(0, series resistance).  A
# negative series resistance -- a transformer-equivalent artefact carried by
# every rung from ACTIVSg10k up -- would make the cut below concave.  See
# `Branch.r_heat` in ropf/network.py.
param r     {e in branches} >= 0 default 0;
param Pd    {i in buses};
param Gs    {i in buses} default 0;
param Pmax  {g in gens};
param Pmin  {g in gens};

# --- cost --------------------------------------------------------------------
param fixedcost {g in gens};
param lincost   {g in gens};
param quadcost  {g in gens};

# --- angle bounds ------------------------------------------------------------
param theta_max {i in buses};
param theta_min {i in buses};
param maxangle  {e in branches};
param minangle  {e in branches};

# --- risk --------------------------------------------------------------------
param risk_weight default 0;      # lambda in (1a); 0 gives the nominal dispatch
# 0 = flow family, eq (6a); 1 = Joule family, eq (6b).  The bus family, eq (6c),
# is carried by its own cut block below and does not use this switch.
param risk_family integer default 0;

# --- variables ---------------------------------------------------------------
var theta {i in buses} >= theta_min[i], <= theta_max[i];
var Pf    {e in branches} >= -U[e], <= U[e];
var Pg    {g in gens} >= Pmin[g], <= Pmax[g];
var Phi   >= 0;                   # the risk surrogate

# --- objective, eq (1a) ------------------------------------------------------
minimize total_cost:
    sum {g in gens} (fixedcost[g] + lincost[g]*Pg[g] + quadcost[g]*Pg[g]^2)
    + risk_weight * Phi;

# --- eq (1b): DC flow definition ---------------------------------------------
subject to Pf_def {e in branches}:
    Pf[e] = bdc[e] * (theta[bus_f[e]] - theta[bus_t[e]]) + Pfinj[e];

# --- eq (1c): active power balance -------------------------------------------
# Written with from-end flows at both ends, as eq (1c) is: the flow into line
# (i,j) at bus i leaves i, and the flow into line (j,i) at bus j arrives at i.
# The to-end flow is not a separate variable because the DC model is lossless
# and the two phase-shift injections cancel, so Pt == -Pf exactly.  That
# identity is asserted by tests/test_matpower_parity.py.
subject to Pbalance {i in buses}:
    (sum {e in branches_f[i]} Pf[e]) - (sum {e in branches_t[i]} Pf[e])
    = (sum {g in bus_gens[i]} Pg[g]) - Pd[i] - Gs[i];

# --- angle difference limits, and the reference bus --------------------------
subject to angle_diff {e in branches}:
    minangle[e] <= theta[bus_f[e]] - theta[bus_t[e]] <= maxangle[e];

###############################################################################
#  Risk cuts, Section 2.4
#
#  Line families.  `choose` holds the branches selected by the separation, and
#  `nCUT` how many are live; both are set from Python, and MAX_CUTS caps the
#  pool.  A cut is appended, never replaced, so the master stays a relaxation
#  of the true functional at every iteration -- eq (11).
###############################################################################

param MAX_CUTS >= 0, integer, default card(branches);
param nCUT >= 0, <= MAX_CUTS, integer, default 0;
param choose {1..MAX_CUTS} in branches;

# eq (6a): the flow family, as the pair of signed inequalities bounding |P_ij|.
subject to FlowCut_pos {k in 1..nCUT: risk_family = 0}:
    Phi >= Pf[choose[k]];
subject to FlowCut_neg {k in 1..nCUT: risk_family = 0}:
    Phi >= -Pf[choose[k]];

# eq (6b): the Joule family, one convex quadratic per selected line.
subject to JouleCut {k in 1..nCUT: risk_family = 1}:
    Phi >= r[choose[k]] * Pf[choose[k]]^2;

###############################################################################
#  eq (6c): the bus family.
#
#  Phi >= sum_{(m,n) in E_i} sigma_mn P_mn + sum_{g in G_i} sigma_g P_g + |P_di|
#
#  The signs are frozen from the incumbent while the flows and unit outputs stay
#  live, which is what makes the cut a subgradient inequality of f_i and hence a
#  minorant of phi^bus.  Because sigma is exactly +1 or -1 (a term whose value
#  is zero at the incumbent contributes nothing and is simply omitted), the
#  cut is carried by four index sets rather than a dense coefficient matrix --
#  which matters at 88,207 branches.
#
#  Declared here rather than built as constraint text from Python, so that the
#  model a reader inspects is the model that is solved, and so the bus cuts are
#  counted the same way the line cuts are.
###############################################################################

param MAX_BUS_CUTS >= 0, integer, default card(buses);
param nBUSCUT >= 0, <= MAX_BUS_CUTS, integer, default 0;

set bus_cut_br_pos  {k in 1..MAX_BUS_CUTS} within branches default {};
set bus_cut_br_neg  {k in 1..MAX_BUS_CUTS} within branches default {};
set bus_cut_gen_pos {k in 1..MAX_BUS_CUTS} within gens     default {};
set bus_cut_gen_neg {k in 1..MAX_BUS_CUTS} within gens     default {};
param bus_cut_demand {k in 1..MAX_BUS_CUTS} >= 0 default 0;   # |P_di|

subject to BusCut {k in 1..nBUSCUT}:
    Phi >= (sum {e in bus_cut_br_pos[k]}  Pf[e])
         - (sum {e in bus_cut_br_neg[k]}  Pf[e])
         + (sum {g in bus_cut_gen_pos[k]} Pg[g])
         - (sum {g in bus_cut_gen_neg[k]} Pg[g])
         + bus_cut_demand[k];

###############################################################################
#
#  (M^ac) -- the AC master problem.
#
#  Section 2.2 of the manuscript, equations (2a)-(2i), minimizing the same
#  objective (1a) on the AC network.  The flow equations are nonconvex, so this
#  is a nonlinear program and a solver returns a local solution.
#
#  Per Section 3.4 this model is solved ONCE, after the cutting-plane loop has
#  run to termination on (M) and its cut pool has been appended here.  Nothing
#  in this file iterates; see ropf/algorithm.py.
#
#  Naming follows the paper: `risk_weight` is lambda, `Phi` the risk surrogate,
#  `theta` a voltage angle, `v` a voltage magnitude.
#
###############################################################################

# --- sets --------------------------------------------------------------------
set buses;
set gens;
set branches;
set branches_f {i in buses};
set branches_t {i in buses};
set bus_gens   {i in buses};

# --- network data ------------------------------------------------------------
param bus_f {e in branches};
param bus_t {e in branches};

# Admittance entries, MATPOWER makeYbus.  See ropf/network.py.
param Gff {e in branches};  param Bff {e in branches};
param Gft {e in branches};  param Bft {e in branches};
param Gtf {e in branches};  param Btf {e in branches};
param Gtt {e in branches};  param Btt {e in branches};

param Gs {i in buses} default 0;
param Bs {i in buses} default 0;
param Pd {i in buses};
param Qd {i in buses};
param U  {e in branches} >= 0;
# The Joule heat coefficient of eq (4c), max(0, series resistance); see
# `Branch.r_heat` in ropf/network.py.
param r  {e in branches} >= 0 default 0;

param Pmax {g in gens};  param Pmin {g in gens};
param Qmax {g in gens};  param Qmin {g in gens};
param Vmax {i in buses} >= 0;
param Vmin {i in buses} >= 0;

param fixedcost {g in gens};
param lincost   {g in gens};
param quadcost  {g in gens};

param theta_max {i in buses};
param theta_min {i in buses};
param maxangle  {e in branches};
param minangle  {e in branches};

# --- starting point ----------------------------------------------------------
param Vinit         {i in buses};
param thetadiffinit {e in branches};

# --- risk --------------------------------------------------------------------
param risk_weight default 0;
param risk_family integer default 0;   # 0 = flow, eq (6a); 1 = Joule, eq (6b)

# --- variables ---------------------------------------------------------------
var v     {i in buses} >= Vmin[i], <= Vmax[i], := Vinit[i];
var theta {i in buses} >= theta_min[i], <= theta_max[i];
var thetadiff {e in branches} >= minangle[e], <= maxangle[e],
                               := thetadiffinit[e];
var Pf {e in branches} >= -U[e], <= U[e];
var Pt {e in branches} >= -U[e], <= U[e];
var Qf {e in branches} >= -U[e], <= U[e];
var Qt {e in branches} >= -U[e], <= U[e];
var Pg {g in gens} >= Pmin[g], <= Pmax[g];
var Qg {g in gens} >= Qmin[g], <= Qmax[g];
var Phi >= 0;

# --- objective, eq (1a) ------------------------------------------------------
minimize total_cost:
    sum {g in gens} (fixedcost[g] + lincost[g]*Pg[g] + quadcost[g]*Pg[g]^2)
    + risk_weight * Phi;

# --- eq (2a)-(2d): flow into each line at each of its two ends ----------------
subject to Pf_def {e in branches}:
    Pf[e] = Gff[e]*v[bus_f[e]]^2
          + v[bus_f[e]]*v[bus_t[e]]*( Gft[e]*cos(thetadiff[e])
                                    + Bft[e]*sin(thetadiff[e]) );

subject to Pt_def {e in branches}:
    Pt[e] = Gtt[e]*v[bus_t[e]]^2
          + v[bus_f[e]]*v[bus_t[e]]*( Gtf[e]*cos(thetadiff[e])
                                    - Btf[e]*sin(thetadiff[e]) );

subject to Qf_def {e in branches}:
    Qf[e] = -Bff[e]*v[bus_f[e]]^2
          + v[bus_f[e]]*v[bus_t[e]]*( Gft[e]*sin(thetadiff[e])
                                    - Bft[e]*cos(thetadiff[e]) );

subject to Qt_def {e in branches}:
    Qt[e] = -Btt[e]*v[bus_t[e]]^2
          - v[bus_f[e]]*v[bus_t[e]]*( Gtf[e]*sin(thetadiff[e])
                                    + Btf[e]*cos(thetadiff[e]) );

# --- eq (2e), (2f): active and reactive balance -------------------------------
# The AC network is lossy, so unlike (M) the to-end flow is a distinct variable
# and both ends must appear.  (Equation (2e) as typeset writes P_ij in both
# sums; the second is the to-end flow.)
subject to Pbalance {i in buses}:
    (sum {e in branches_f[i]} Pf[e]) + (sum {e in branches_t[i]} Pt[e])
    = (sum {g in bus_gens[i]} Pg[g]) - Pd[i] - Gs[i]*v[i]^2;

subject to Qbalance {i in buses}:
    (sum {e in branches_f[i]} Qf[e]) + (sum {e in branches_t[i]} Qt[e])
    = (sum {g in bus_gens[i]} Qg[g]) - Qd[i] + Bs[i]*v[i]^2;

# --- eq (2g): apparent power limit at both ends -------------------------------
subject to limit_f {e in branches}: Pf[e]^2 + Qf[e]^2 <= U[e]^2;
subject to limit_t {e in branches}: Pt[e]^2 + Qt[e]^2 <= U[e]^2;

# --- eq (2i): angle difference ------------------------------------------------
subject to thetadiff_def {e in branches}:
    thetadiff[e] = theta[bus_f[e]] - theta[bus_t[e]];

###############################################################################
#  Risk cuts, Section 2.4.  Identical in form to (M): the cut pool transferred
#  from the DC stage is appended here unchanged, which is what makes the hybrid
#  of Section 3.4 a transfer rather than a re-separation.
###############################################################################

param MAX_CUTS >= 0, integer, default card(branches);
param nCUT >= 0, <= MAX_CUTS, integer, default 0;
param choose {1..MAX_CUTS} in branches;

subject to FlowCut_pos {k in 1..nCUT: risk_family = 0}:
    Phi >= Pf[choose[k]];
subject to FlowCut_neg {k in 1..nCUT: risk_family = 0}:
    Phi >= -Pf[choose[k]];

subject to JouleCut {k in 1..nCUT: risk_family = 1}:
    Phi >= r[choose[k]] * Pf[choose[k]]^2;

# eq (6c), the bus family.  Each incident line contributes the flow at its own
# from-end, so this reads Pf at both endpoints and never Pt -- the same
# convention as the functional in eq (4a).
param MAX_BUS_CUTS >= 0, integer, default card(buses);
param nBUSCUT >= 0, <= MAX_BUS_CUTS, integer, default 0;

set bus_cut_br_pos  {k in 1..MAX_BUS_CUTS} within branches default {};
set bus_cut_br_neg  {k in 1..MAX_BUS_CUTS} within branches default {};
set bus_cut_gen_pos {k in 1..MAX_BUS_CUTS} within gens     default {};
set bus_cut_gen_neg {k in 1..MAX_BUS_CUTS} within gens     default {};
param bus_cut_demand {k in 1..MAX_BUS_CUTS} >= 0 default 0;

subject to BusCut {k in 1..nBUSCUT}:
    Phi >= (sum {e in bus_cut_br_pos[k]}  Pf[e])
         - (sum {e in bus_cut_br_neg[k]}  Pf[e])
         + (sum {g in bus_cut_gen_pos[k]} Pg[g])
         - (sum {g in bus_cut_gen_neg[k]} Pg[g])
         + bus_cut_demand[k];

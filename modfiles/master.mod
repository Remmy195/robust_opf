# (M): DC master (eq:obj to eq:genlim) with the exposure cuts of eq:cuts.

set buses;
set gens;
set branches;
set branches_f {i in buses};
set branches_t {i in buses};
set bus_gens   {i in buses};

param bus_f {e in branches};
param bus_t {e in branches};
param bdc   {e in branches};
param Pfinj {e in branches} default 0;
param U     {e in branches} >= 0;
param r     {e in branches} >= 0 default 0;
param Pd    {i in buses};
param Gs    {i in buses} default 0;
param Pmax  {g in gens};
param Pmin  {g in gens};

param fixedcost {g in gens};
param lincost   {g in gens};
param quadcost  {g in gens};

param theta_max {i in buses};
param theta_min {i in buses};
param maxangle  {e in branches};
param minangle  {e in branches};

param risk_weight default 0;
param risk_family integer default 0;   # 0 flow, 1 Joule

var theta {i in buses} >= theta_min[i], <= theta_max[i];
var Pf    {e in branches} >= -U[e], <= U[e];
var Pg    {g in gens} >= Pmin[g], <= Pmax[g];
var Phi   >= 0;

minimize total_cost:
    sum {g in gens} (fixedcost[g] + lincost[g]*Pg[g] + quadcost[g]*Pg[g]^2)
    + risk_weight * Phi;

subject to Pf_def {e in branches}:
    Pf[e] = bdc[e] * (theta[bus_f[e]] - theta[bus_t[e]]) + Pfinj[e];

# Lossless, so Pt = -Pf.
subject to Pbalance {i in buses}:
    (sum {e in branches_f[i]} Pf[e]) - (sum {e in branches_t[i]} Pf[e])
    = (sum {g in bus_gens[i]} Pg[g]) - Pd[i] - Gs[i];

subject to angle_diff {e in branches}:
    minangle[e] <= theta[bus_f[e]] - theta[bus_t[e]] <= maxangle[e];

param q integer >= 1 default 1;
param mean_form binary default 0;

param MAX_CUTS >= 0, integer, default card(branches);
param nCUT >= 0, <= MAX_CUTS, integer, default 0;
param choose {1..MAX_CUTS} in branches;

# Flow cuts, q = 1.
subject to FlowCut_pos {k in 1..nCUT: risk_family = 0 and mean_form = 0}:
    Phi >= Pf[choose[k]];
subject to FlowCut_neg {k in 1..nCUT: risk_family = 0 and mean_form = 0}:
    Phi >= -Pf[choose[k]];

# Joule cuts, q = 1.
subject to JouleCut {k in 1..nCUT: risk_family = 1 and mean_form = 0}:
    Phi >= r[choose[k]] * Pf[choose[k]]^2;

param MAX_BUS_CUTS >= 0, integer, default card(buses);
param nBUSCUT >= 0, <= MAX_BUS_CUTS, integer, default 0;

set bus_cut_br_pos  {k in 1..MAX_BUS_CUTS} within branches default {};
set bus_cut_br_neg  {k in 1..MAX_BUS_CUTS} within branches default {};
param bus_cut_bus   {k in 1..MAX_BUS_CUTS} in buses;

# Bus cuts, q = 1.
subject to BusCut {k in 1..nBUSCUT: mean_form = 0}:
    Phi >= (sum {e in bus_cut_br_pos[k]}  Pf[e])
         - (sum {e in bus_cut_br_neg[k]}  Pf[e]);

# eq:cutmean, for q > 1: one u per pooled line and per pooled bus.
set BUS_POOL = setof {k in 1..nBUSCUT} bus_cut_bus[k];

var t_mean {1..mean_form} >= 0;
var u_line {k in 1..nCUT: mean_form = 1} >= 0;
var u_bus  {i in BUS_POOL: mean_form = 1} >= 0;

subject to MeanCut {z in 1..mean_form}:
    Phi >= t_mean[z] + (1/q) * (sum {k in 1..nCUT} u_line[k]
                                + sum {i in BUS_POOL} u_bus[i]);

subject to FlowExcess_pos {k in 1..nCUT, z in 1..mean_form: risk_family = 0}:
    u_line[k] >= Pf[choose[k]] - t_mean[z];
subject to FlowExcess_neg {k in 1..nCUT, z in 1..mean_form: risk_family = 0}:
    u_line[k] >= -Pf[choose[k]] - t_mean[z];

subject to JouleExcess {k in 1..nCUT, z in 1..mean_form: risk_family = 1}:
    u_line[k] >= r[choose[k]] * Pf[choose[k]]^2 - t_mean[z];

subject to BusExcess {k in 1..nBUSCUT, z in 1..mean_form}:
    u_bus[bus_cut_bus[k]] >= (sum {e in bus_cut_br_pos[k]}  Pf[e])
                           - (sum {e in bus_cut_br_neg[k]}  Pf[e]) - t_mean[z];

# (M^ac): AC master (eq:ac:pf to eq:ac:anglelim), objective eq:obj, exposure cuts at q = 1.

set buses;
set gens;
set branches;
set branches_f {i in buses};
set branches_t {i in buses};
set bus_gens   {i in buses};

param bus_f {e in branches};
param bus_t {e in branches};

param Gff {e in branches};  param Bff {e in branches};
param Gft {e in branches};  param Bft {e in branches};
param Gtf {e in branches};  param Btf {e in branches};
param Gtt {e in branches};  param Btt {e in branches};

param Gs {i in buses} default 0;
param Bs {i in buses} default 0;
param Pd {i in buses};
param Qd {i in buses};
param U  {e in branches} >= 0;
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

param Vinit         {i in buses};
param thetadiffinit {e in branches};

param risk_weight default 0;
param risk_family integer default 0;   # 0 flow, 1 Joule

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

minimize total_cost:
    sum {g in gens} (fixedcost[g] + lincost[g]*Pg[g] + quadcost[g]*Pg[g]^2)
    + risk_weight * Phi;

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

subject to Pbalance {i in buses}:
    (sum {e in branches_f[i]} Pf[e]) + (sum {e in branches_t[i]} Pt[e])
    = (sum {g in bus_gens[i]} Pg[g]) - Pd[i] - Gs[i]*v[i]^2;

subject to Qbalance {i in buses}:
    (sum {e in branches_f[i]} Qf[e]) + (sum {e in branches_t[i]} Qt[e])
    = (sum {g in bus_gens[i]} Qg[g]) - Qd[i] + Bs[i]*v[i]^2;

subject to limit_f {e in branches}: Pf[e]^2 + Qf[e]^2 <= U[e]^2;
subject to limit_t {e in branches}: Pt[e]^2 + Qt[e]^2 <= U[e]^2;

subject to thetadiff_def {e in branches}:
    thetadiff[e] = theta[bus_f[e]] - theta[bus_t[e]];

param MAX_CUTS >= 0, integer, default card(branches);
param nCUT >= 0, <= MAX_CUTS, integer, default 0;
param choose {1..MAX_CUTS} in branches;

subject to FlowCut_pos {k in 1..nCUT: risk_family = 0}:
    Phi >= Pf[choose[k]];
subject to FlowCut_neg {k in 1..nCUT: risk_family = 0}:
    Phi >= -Pf[choose[k]];

subject to JouleCut {k in 1..nCUT: risk_family = 1}:
    Phi >= r[choose[k]] * Pf[choose[k]]^2;

param MAX_BUS_CUTS >= 0, integer, default card(buses);
param nBUSCUT >= 0, <= MAX_BUS_CUTS, integer, default 0;

set bus_cut_br_pos  {k in 1..MAX_BUS_CUTS} within branches default {};
set bus_cut_br_neg  {k in 1..MAX_BUS_CUTS} within branches default {};

subject to BusCut {k in 1..nBUSCUT}:
    Phi >= (sum {e in bus_cut_br_pos[k]}  Pf[e])
         - (sum {e in bus_cut_br_neg[k]}  Pf[e]);

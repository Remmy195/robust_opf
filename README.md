# ropf — risk-aware optimal power flow by cutting planes

Companion code for the manuscript *Risk-aware optimal power flow*. It implements
Algorithm 1 of Section 3, the three risk functionals of Section 2.3, the AC
stage of Section 3.4, and the counterfactual campaign of Section 4, and it runs
them over the six-instance ACTIVSg ladder of Section 5.

Every optimization model in this repository is an AMPL `.mod` file under
[modfiles/](modfiles/). There is no model built in Python.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,study]"
```

Requires Python 3.10+, an AMPL installation with a valid licence, and a solver.
The study uses Gurobi for the linear models — (M) and (D) — and Knitro for the
nonlinear AC master (M^ac); Ipopt works in place of Knitro at the smaller instances.

## The test systems

The six instances are the Texas A&M ACTIVSg synthetic grids. They are **not**
committed: the distributions run to about 875 MB. One small case,
`case_ACTIVSg200.m`, is tracked under [tests/fixtures/](tests/fixtures/) so the
test suite runs on a bare clone.

Texas A&M serves them from a landing page that requires accepting terms, so
they have to be downloaded by hand. Unpack the six archives into `data/`:

```bash
cd data
for z in ~/ACTIVSg/ACTIVSg*.zip; do
    unzip -j -o "$z" 'case_ACTIVSg*.m' '*.dyr' '*.aux'
done
```

That covers five instances; ACTIVSg70k ships as a directory rather than a zip, so
copy the same three files out of it. It also pulls in a few `*_contingencies.aux`
and `*_dynamics.aux` the study does not read, which are harmless.

Nothing renames anything. The reader already knows that ACTIVSg25k calls its
dynamics file `ACTIVSg25k.dyr` where the other five use
`ACTIVSg<n>_dynamics.dyr`, and reads either.

Point `dynamics_search` at `data` and the study is self-contained. It can also
be pointed straight at the directory holding the archives — the `.dyr` and
`.aux` are read out of a zip without unpacking it — in which case only the
`.m` cases need to be extracted.

Which case a result came from is recorded per run, not asserted up front:
`solution_summary.json` carries the SHA-256 of the case that run actually read.
The ACTIVSg distributions are not versioned, so that digest is what makes "the
same case" checkable months later.

## The three functionals

A run picks one. The config key is `metric`.

| `metric` | eq | what it measures |
| --- | --- | --- |
| `max_active_flow` | (4b) | `max_e \|P_e\|` — the loading of the most heavily used line |
| `joule_loss_max` | (4c) | `max_e r_e P_e²` — the ohmic heating of the worst line |
| `bus_flow_sum_agg` | (4a) | `max_i Σ_{(m,n)∈E_i} \|P_mn\|` — the power crossing the busiest bus |

**φ^bus is the incident line flows and nothing else** — no generation term and
no demand term. Both omissions are deliberate and worth stating, because the
obvious reading of "the power at a bus" includes them:

* **generation is already in the flows.** Kirchhoff at bus *i* says what a unit
  injects there leaves through the incident lines, so a `Σ_g |P_g|` term counts
  the same power twice and weights a generator bus against a transit bus by
  where the metering happens rather than by how much power moves;
* **demand is fixed data.** `P_di` is identical at every dispatch and every
  λ, so it cannot be traded against anything. Carried in `f_i` it only adds a
  per-bus offset that reorders the argmax, making the functional report the most
  heavily *loaded* bus rather than the busiest one, and putting a constant into
  the cut that the master can never move.

`f_i` sums magnitudes, so flows do not cancel: a degree-2 bus carrying *P*
through it scores `2|P|`, which is correct rather than a symptom — the power
crosses two lines.

The signs enter only in the cut, frozen at the incumbent. That is what makes
eq (6c) a subgradient inequality of `f_i`, hence a minorant of φ^bus, hence
eq (11).

## Running a solve

```bash
ropf solve configs/activs200_flow_baseline.conf
```

This sweeps the weight grid for one (case, metric, stage) and writes into
`results/<tag>/`:

| file | what it holds |
| --- | --- |
| `solution_summary.json` | every run in full: dispatch, per-iteration trace, provenance |
| `efficient_frontier.csv` | one row per weight — cost, risk, the Gamma gap |
| `efficient_frontier.json` | the same rows, with the dominated ones marked |
| `dispatch_bus.csv` | every bus at every weight |
| `dispatch_gen.csv` | every unit at every weight, and its move from nominal |
| `dispatch_branch.csv` | every branch at every weight: flow, loading, and the two bad-line flags |

`run.log` beside them is the transcript. Nothing reads it back.

### Which lines are bad

Two different questions, so two columns in `dispatch_branch.csv` and two lines
in the transcript at every iteration:

* **`high_exposure`** — the component carries at least `exposure_fraction` of
  ρ, the run's own functional. Relative to what the run is minimizing, so under
  `bus_flow_sum_agg` it flags buses (in `dispatch_bus.csv`) rather than lines.
  `exposure_fraction = 1` is the argmax alone; the default is 0.95.
* **`overloaded`** — `|P_e|` is at or above the branch's own `rateA`. Absolute,
  and independent of the run's functional. Branches the case gives no rating
  carry the big-M substitution and are never flagged.

`efficient_frontier.csv` carries the counts, `n_exposed` and `n_overloaded`,
one row per weight.

### LP files, one per iteration

`write_lp = true` writes the problem of every master solve to
`results/<tag>/lp/w<λ>/master_k<k>.lp`, labelled by the Algorithm 1 iteration
count — `master_k000.lp` is the nominal solve of line 1. One directory per
weight, since two weights sharing one would overwrite each other's `k`.

The solver writes the file during the solve, so this costs no extra solve; it
does cost one file per iteration per weight. It needs a DC solver whose driver
can write a problem file (Gurobi and the other simplex/barrier drivers can;
Knitro cannot, and LP output is refused with a note rather than silently
skipped). Rows and columns carry solver-generated names — AMPL's own entity
names are not exposed through the driver — so the LP is for inspecting the
model's shape and size, and the flags above are what name components.

### The configuration file

A run is described by a text file of `key = value` lines. `ropf keys` prints
every key with its type and default.

**The config file is the only place a study parameter is set.** No command-line
flag overrides a value in it — the two flags that exist, `--quiet` and
`--dry-run`, change what is printed and whether anything is solved, and neither
can change a number. An override would mean the same file producing two
different studies depending on how it was invoked, and the output directory
could no longer be read as a record of what was run.

Two kinds of mistake are errors rather than warnings:

* an **unknown key**, because a config that ignores what it does not recognise
  turns a typo into a study that measured the wrong thing, and leaves no trace
  of it in the output;
* a **duplicate key**, because there is no defensible answer to which of the two
  values was meant. Last-one-wins is the usual choice and the worst one.

Use `--dry-run` to see the effective configuration without solving.

### The three stages

| stage | what runs |
| --- | --- |
| `baseline` | Algorithm 1 on (M); the reported dispatch is the DC one |
| `a2` | Algorithm 1 on (M), then (M^ac) solved **once** with the pool the DC loop accumulated (Section 3.4) |
| `a3` | Algorithm 1 on (M^ac) throughout, separating on the AC dispatch |

The AC stage of `a2` is one solve, and there is no knob that says otherwise:
Section 3.4 is a transfer, not a re-separation.

## The ladder study

Section 5 runs the whole grid: six instances × three functionals × three stages,
and the Section 4 counterfactual campaign against every dispatch each of those
produces.

```bash
ropf ladder configs/ladder.conf --dry-run   # what would run, and how big
ropf ladder configs/ladder.conf             # claim one combo and run it
ropf ladder configs/ladder.conf --all       # keep going until none left
ropf ladder configs/ladder.conf --status    # the progress table
```

The unit of work is one **combo** — one (instance, metric, stage) triple. One
invocation claims one combo, runs its frontier and its campaign, writes
`campaign.json`/`campaign.csv` beside the three frontier artifacts, and marks
the directory `DONE`. Because the combos are independent, that gives both
parallelism and resume for free: run the command in as many shells as you have
cores, and run it again after a crash. A combo is claimed by creating
`.claim/` inside its directory, which is atomic, so two processes racing for
the same combo cannot both get it.

Each dispatch is evaluated against two classes of disfigurement (Section 4.1) —
`gen_k`, disabling the K highest-output units, and `walk_k`/`walk_draws`,
random-walk component sets — and each survivor is redispatched by model (D)
after clearing the frequency screen.

Two things the campaign fixes once and never revisits, because varying them
would make the dispatches incomparable:

* **the top-K ranking**, taken from the nominal dispatch at lambda = 0.
  Re-ranking per dispatch would flatter the de-risked ones, whose top-K carries
  less power by construction;
* **the disfigurement list**, built once per combo from a declared seed, so
  every dispatch faces the same disturbances.

### What the campaign needs that the cases do not carry

The frequency screen of Section 4.2 needs each instance's PSS/E `.dyr` (and `.aux`
for AGC participation). These ship inside the same archives as the cases, so
unpacking as above puts them beside the cases and `dynamics_search = data` is
correct.

Its four constants — `f0_hz`, `rocof_max_hz_s`, `f_under_hz` and the load
damping — are **not defaulted by the code**, and the config file inherits that
refusal. They are grid-code quantities: supplying a plausible number would mean
reporting it as if it were data. The damping must additionally name its base,
because the textbook 1-2 %/% figure is on the *load* base and the swing
equation needs the *system* base; the two differ by the load-to-baseMVA ratio
(671 on ACTIVSg2000) and getting it wrong fails quietly, with the nadir simply
coming out at the wrong depth.

`beta`, the emergency rating factor of eq (6e), **defaults to 1, which is the
case authors' own rule.** The per-branch emergency *magnitude* really is
missing -- `rateB` and `rateC` are zero across every ACTIVSg `.m`, `.aux` and
`.RAW` -- but the monitoring *rule* is present: all six `.aux` files carry the
same `LimitSet` record, rate set "A" for the base case and "A" again for the
contingency case at `LSLinePercent 100` (`data/ACTIVSg2000.aux:1281`). So the
cited default is 1 and the burden of declaration falls on 1.2, the 20%
short-term overload, which stays in the sweep as a labelled assumption. See
`COUNTERFACTUAL_STANDARD.md` sections 3.5 and 5.3.

Sweeping it costs nothing. `campaign.csv` carries `worst_loading`,
`max |Pf| / rateA` over rated surviving branches, which does not mention `beta`,
so one run is read at every rating factor. Nothing forbids the overload any
more either: the post-event rating is soft, carrying GO3's slack `s_jtk^+`, so
a draw whose flows exceed `beta * rateA` reports `overload_max_pu` and
`overload_sum_pu` instead of coming back infeasible.

## Scoring a frontier that already exists

`ropf ladder` can only score a dispatch it just computed, so it scores whatever
`weight_grid` its config names. `ropf score` runs the same Section 4 campaign
against dispatches that are already on disk.

```bash
ropf score configs/frontier_campaign.conf --dry-run  # what would be scored, at what overhead
ropf score configs/frontier_campaign.conf --all      # keep going until none left
ropf score configs/frontier_campaign.conf --status   # the progress table
```

**Algorithm 1 does not run.** That is the command, not an optimisation of it: a
second sweep would produce a second set of dispatches, and the point is to
score *these*. The tree named by `frontier_dir` is read-only — `dispatch_gen
.csv`, `efficient_frontier.csv` and `solution_summary.json` are read out of it
and nothing is written back — and the campaign goes to `outdir` under the
ladder's own combo names, so one reader reads both kinds of tree.
`source.json` beside each campaign records which frontier directory it came
from and, per weight, that sweep's termination, cost and overhead.

**Why it exists: on cost overhead, a multiplier grid bunches.** A multiplier is
an exchange rate, not an operating point. Measured on
`results/vendor_gamma0_soft/`, five of the six ACTIVSg200 `max_active_flow`
multipliers buy the identical 9.04% overhead and the whole ACTIVSg2000 sweep
spans about one point, so a security claim indexed by multiplier is made at
operating points nobody would run at. The `frontier_v1` sweeps, whose grids
were bisected per combo around λ\*, land at 1.57, 1.63, 1.70, 1.99, 2.00% and
on out to 26. `campaign.csv` now carries `cost_overhead` on every row for
exactly this reason, and `analysis/vendor_report.py` leads with it.

A score config sets no Algorithm 1 parameter — `weight_grid`, `kappa`, `eta`,
`k_bar` and `solver_ac` are not keys of one, and naming any of them is an
unknown-key error rather than a value that would go unread. The weights scored
are the ones each source sweep left behind, at the λ\* it measured.

### Threads

Set `threads` to the **physical** core count, not the logical one. On a
2-socket box with 16 physical cores and 32 logical CPUs, letting Gurobi take
its default cost 5.7× on a single model (D) evaluation at the 70k instance — 64.7s
against 11.4s — for an answer agreeing to nine significant figures.

## Layout

```
src/ropf/
  network.py        MATPOWER case files to the network model
  risk.py           the functionals of Section 2.3, their separation, and the
                    exposure and overload flags
  model.py          the AMPL boundary: (M), (M^ac), (D), and LP output
  algorithm.py      Algorithm 1 and the Section 3.4 AC stage
  config.py         the configuration file and its key table
  results.py        the run artifacts
  log.py            the run transcript
  counterfactual/   Section 4: disfigurements, the frequency screen, (D)
  study/frontier.py the adaptive frontier tracer behind `ropf trace`
  study/ladder.py   both study drivers: combos, claims, the campaign, and the
                    path that scores a frontier it did not compute
modfiles/           master.mod, master_ac.mod, postevent.mod
```

`model.py` is the only module that touches AMPL, and it owns the entity
lifecycle rule. Every line of Algorithm 1 is tagged `ALG1-Ln` in
`algorithm.py` at the code implementing it, so a line of the pseudocode with no
code under it — or code under no line — is visible in one screenful.

## Tests

```bash
pytest
```

The suite anchors the reader against MATPOWER itself.
[tests/fixtures/dump_matpower.m](tests/fixtures/dump_matpower.m) reads
MATPOWER 8.1's own assembled `makeYbus` and `makeBdc` matrices and imports
nothing from `ropf`, so the parity test is non-circular. Regenerate the fixture
with Octave:

```bash
cd tests/fixtures
OCTAVE_HOME=$HOME/miniconda3 $OCTAVE_HOME/bin/octave-cli --no-gui \
    dump_matpower.m case_ACTIVSg200.m matpower_ACTIVSg200.csv
```

It takes the case and the output file as arguments. Octave here is the
miniconda build and needs `OCTAVE_HOME` set, or every core function is
undefined — which looks like a MATPOWER problem and is not one.

Tests that need AMPL skip cleanly when it is absent.

## Licence

See [LICENSE](LICENSE).

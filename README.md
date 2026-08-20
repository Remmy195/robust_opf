# ropf — risk-aware optimal power flow by cutting planes

Companion code for the manuscript *Risk-aware optimal power flow*. It implements
Algorithm 1 of Section 3, the three risk functionals of Section 2.3, the AC
stage of Section 3.4, and the counterfactual campaign of Section 4, and it runs
them over the six-rung ACTIVSg ladder of Section 5.

Every optimization model in this repository is an AMPL `.mod` file under
[modfiles/](modfiles/). There is no model built in Python.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,study]"
```

Requires Python 3.10+, an AMPL installation with a valid licence, and a solver.
The study uses Gurobi for the linear models — (M) and (D) — and Knitro for the
nonlinear AC master (M^ac); Ipopt works in place of Knitro at the smaller rungs.

## The test systems

The six rungs are the Texas A&M ACTIVSg synthetic grids. They are **not**
committed: the distributions run to about 875 MB. One small case,
`case_ACTIVSg200.m`, is tracked under [tests/fixtures/](tests/fixtures/) so the
test suite runs on a bare clone.

Texas A&M serves them from a landing page that requires accepting terms, so
nothing can be retrieved unattended. Download the six distributions by hand,
then adopt them:

```bash
ropf fetch --list           # the six rungs
ropf fetch --from ~/ACTIVSg # take cases AND dynamics from the archives
ropf fetch                  # verify what is present, report what is missing
```

`--from` reads the distributions as TAMU ships them -- `ACTIVSg200.zip` and the
like, or an unpacked directory of the same name -- and writes three files per
rung into `data/`: the MATPOWER case, the `.dyr` machine records the frequency
screen needs, and the `.aux` participation factors. The dynamics ship *inside
the same archives as the cases*, so taking both at once is what makes `data/`
self-contained; a tree built from case files alone leaves Section 4.2 with
nothing to read.

Every rung declares the SHA-256 of the case it must produce, and **it is checked
before anything is written**. The ACTIVSg distributions are not versioned, so
two archives can hold different vintages of `case_ACTIVSg<n>.m` under the same
name -- without the digest a study could be reproduced against a different case
and report it as the same one, and without checking *first* one command could
silently swap the case an existing tree was built on. A mismatched case is
reported and skipped, leaving the existing file alone. Run `ropf fetch` with no
arguments over an existing `data/` to check it.

## Running a solve

```bash
ropf solve configs/activs200_flow_baseline.conf
```

This sweeps the weight grid for one (case, metric, stage) and writes three
artifacts into `results/<tag>/`:

| file | what it holds |
| --- | --- |
| `solution_summary.json` | every run in full: dispatch, per-iteration trace, provenance |
| `efficient_frontier.csv` | one row per weight — cost, risk, the Gamma gap |
| `efficient_frontier.json` | the same rows, with the dominated ones marked |

A fourth file, `run.log`, is the transcript. Nothing reads it back.

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

Section 5 runs the whole grid: six rungs x three functionals x three stages,
and the Section 4 counterfactual campaign against every dispatch each of those
produces.

```bash
ropf ladder configs/ladder.conf --dry-run   # what would run, and how big
ropf ladder configs/ladder.conf             # claim one combo and run it
ropf ladder configs/ladder.conf --all       # keep going until none left
ropf ladder configs/ladder.conf --status    # the progress table
```

The unit of work is one **combo** -- one (rung, metric, stage) triple. One
invocation claims one combo, runs its frontier and its campaign, writes
`campaign.json`/`campaign.csv` beside the three frontier artifacts, and marks
the directory `DONE`. Because the combos are independent, that gives both
parallelism and resume for free: run the command in as many shells as you have
cores, and run it again after a crash. A combo is claimed by creating
`.claim/` inside its directory, which is atomic, so two processes racing for
the same combo cannot both get it.

Two things the campaign fixes once and never revisits, because varying them
would make the dispatches incomparable:

* **the top-K ranking**, taken from the nominal dispatch at lambda = 0.
  Re-ranking per dispatch would flatter the de-risked ones, whose top-K carries
  less power by construction;
* **the disfigurement list**, built once per combo from a declared seed, so
  every dispatch faces the same disturbances.

### What the campaign needs that the cases do not carry

The frequency screen of Section 4.2 needs each rung's PSS/E `.dyr` (and `.aux`
for AGC participation). `ropf fetch --from` puts both in `data/` beside the
cases, under the names the reader looks for -- including for ACTIVSg25k, which
ships its `.dyr` without the `_dynamics` the other five use -- so
`dynamics_search = data` is correct and needs no path outside the repository.

Its four constants -- `f0_hz`, `rocof_max_hz_s`, `f_under_hz` and the load
damping -- are **not defaulted by the code**, and the config file inherits that
refusal. They are grid-code quantities: supplying a plausible number would mean
reporting it as if it were data. The damping must additionally name its base,
because the textbook 1-2 %/% figure is on the *load* base and the swing
equation needs the *system* base; the two differ by the load-to-baseMVA ratio
(671 on ACTIVSg2000) and getting it wrong fails quietly, with the nadir simply
coming out at the wrong depth.

### Threads

Set `threads` to the **physical** core count, not the logical one. On a
2-socket box with 16 physical cores and 32 logical CPUs, letting Gurobi take
its default cost 5.7x on a single model (D) evaluation at the 70k rung -- 64.7s
against 11.4s -- for an answer agreeing to nine significant figures.

## Layout

```
src/ropf/
  network.py        MATPOWER case files to the network model
  risk.py           the functionals of Section 2.3 and their separation
  model.py          the AMPL boundary: (M), (M^ac), (D)
  algorithm.py      Algorithm 1 and the Section 3.4 AC stage
  config.py         the configuration file and its key table
  results.py        the three artifacts
  log.py            the run transcript
  counterfactual/   Section 4: disfigurements, the frequency screen, (D)
  study/ladder.py   the ladder study driver: combos, claims, the campaign
  data/             fetching and verifying the ACTIVSg cases
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

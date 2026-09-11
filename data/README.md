# Case data

The six ladder instances are TAMU synthetic grids and are **not** committed: the
full distributions are about 875 MB. Texas A&M serves them from a landing page
that requires accepting terms, so they are downloaded and unpacked by hand into
this directory — see the [top-level README](../README.md) for the exact
commands.

Each distribution carries more than the `.m` case file. The `.dyr` gives machine
inertia and governor droop, and the `.aux` gives generator MVA bases, AGC
participation factors and the vendor N-1 contingency list; Section 4's frequency
screen, post-event model and class (c) need all three, so keep the whole
distribution rather than extracting the case alone.

Two irregularities in how these are published are handled by
`ropf.counterfactual.dynamics.locate` and should not be "tidied" away:

* `ACTIVSg25k` names its dynamics file `ACTIVSg25k.dyr`, not
  `ACTIVSg25k_dynamics.dyr` as every other instance does.
* `ACTIVSg70k` ships as an unpacked directory rather than a zip archive.

`tests/fixtures/case_ACTIVSg200.m` is committed, at 56 KB, so that the MATPOWER
parity test runs on a fresh clone with no download.

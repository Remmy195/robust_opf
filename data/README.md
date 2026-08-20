# Case data

The six ladder rungs are TAMU synthetic grids and are **not** committed: the
full distributions are about 875 MB. Fetch them with

```bash
python data/fetch.py --rung activs200      # one rung
python data/fetch.py --all                 # the whole ladder
```

Each distribution carries more than the `.m` case file. The `.dyr` gives machine
inertia and governor droop, and the `.aux` gives generator MVA bases and AGC
participation factors; Section 4's frequency screen and post-event model need
both, so the fetch keeps the whole distribution rather than extracting the case
alone.

Two irregularities in how these are published are handled by `fetch.py` and
should not be "tidied" away:

* `ACTIVSg25k` names its dynamics file `ACTIVSg25k.dyr`, not
  `ACTIVSg25k_dynamics.dyr` as every other rung does.
* `ACTIVSg70k` ships as an unpacked directory rather than a zip archive.

`tests/fixtures/case_ACTIVSg200.m` is committed, at 56 KB, so that the MATPOWER
parity test runs on a fresh clone with no download.

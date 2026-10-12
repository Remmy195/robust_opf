# ropf

Risk-aware optimal power flow by cutting planes. Implements Algorithm 1 of
*Risk-aware optimal power flow* on the DC master (M), and the hybrid stage that
solves the AC master (M^ac) once on the DC cut pool.

## Requirements

- Python 3.10 or later
- AMPL with a licence
- Gurobi for (M); Knitro or Ipopt for (M^ac)

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
ropf configs/risk_aware.conf
```

Writes `result.json` (config, iterates, dispatch), `result.csv` (iterates),
`dispatch_bus.csv`, `dispatch_branch.csv`, `dispatch_gen.csv` and `run.log` to
`results/<case>_<metric>_<stage>/`. A config is `key = value` lines; unknown or
repeated keys are errors.

| Key | Default | Meaning |
|---|---|---|
| `case` | | MATPOWER file; a bare name is looked up in `data/` |
| `metric` | `max_active_flow` | exposure representation: `max_active_flow`, `joule_loss_max`, `bus_flow_sum_agg` |
| `stage` | `baseline` | `a2` adds the hybrid AC solve |
| `flow_domain` | `all` | `rated` drops branches with no rateA from the representation |
| `q` | `1` | order: mean of the q largest contributions |
| `weight_multiplier` | `1` | exposure weight as a multiple of lambda_ref = z0/rho0; `0` gives the plain OPF |
| `risk_weight` | | absolute exposure weight; overrides `weight_multiplier` |
| `kappa` | `5` | cuts per iteration |
| `eta` | `0.85` | stop when rho/rho0 <= 1 - eta |
| `k_bar` | `40` | iteration limit |
| `gamma_tol` | `1e-6` | stop when the gap Gamma <= gamma_tol |
| `solver_dc`, `solver_ac` | `gurobi`, `knitro` | |
| `solver_dc_options`, `solver_ac_options` | | extra AMPL solver options, e.g. `numericfocus=3` |
| `time_limit_s` | `3600` | per solve |
| `threads` | solver default | |
| `solver_verbose` | `false` | |
| `write_lp` | `false` | write each master solve to `lp/` as an LP file; needs a driver with an LP writer, which the AC solvers are not |
| `outdir` | | output directory |

## Python

```python
from ropf import AlgorithmConfig, read_matpower, run

net = read_matpower("data/case_ACTIVSg200.m")
result = run(net, AlgorithmConfig(metric="joule_loss_max"))
result.final.rho, result.dispatch.Pg
```

Sweeping `weight_multiplier` traces the cost-exposure frontier; see below.

## Analysis

`analysis/frontier.py` traces the cost-exposure frontier, running Algorithm 1
once per exposure weight. Its settings are constants at the top of the file:
the keys of a config plus the weights to trace. Edit them and run:

```bash
python analysis/frontier.py
```

Writes `frontier.csv`, `frontier.json`, `frontier.png` and `run.log` to
`results/<case>_<metric>_<stage>_frontier/`. The plot needs matplotlib, which
`ropf` itself does not.

`analysis/xeval.py` cross-evaluates the representations: it solves under each
one at a single weight, then scores every resulting dispatch, and the plain
OPF, under all of them. The solves are the whole cost, since scoring a
representation on a dispatch is arithmetic on Pf. Read the matrix by column: the
lowest entry should be the diagonal, and an off-diagonal entry above 1 means
de-risking under one representation raised what another names. Writes
`xeval.csv`, `xeval.json`, `xeval.png` and `run.log` to `results/<case>_xeval/`.

## Data

`data/case_ACTIVSg200.m` ships with the repo. [data/README.md](data/README.md)
lists the cases the paper uses and where to get them.

## Tests

```bash
pytest
```

Tests that solve skip without AMPL and Gurobi.

## Licence

MIT. See [LICENSE](LICENSE).

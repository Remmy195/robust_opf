# ropf

`ropf` computes a de-risked dispatch: a DC optimal power flow whose objective adds an exposure surrogate $\Phi$, priced at a weight $\lambda$, to the generation cost. An iterative cutting-plane method after Bienstock and Sisson builds $\Phi$ from the components of largest contribution, and an optional AC stage solves the AC master once on the DC cut pool.

## Installation

`ropf` needs Python 3.10 or later and AMPL with a licence: Gurobi for the DC master (M), Knitro or Ipopt for the AC master (M^ac).

```bash
pip install -e ".[dev]"
```

## How the documentation is structured

- [Tutorials](tutorials/index.md) run the package step by step on ACTIVSg200.
- [How-to guides](howto/index.md) answer one question each.
- [Explanation](explanation/index.md) derives the method.
- [Examples](examples/index.md) are worked studies on larger systems.
- [API reference](reference.md) lists the public functions and classes.

## Citing ropf

See [Citing](citing.md).

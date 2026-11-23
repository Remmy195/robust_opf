# The AC stage

!!! note "Draft"
    Convert to a notebook once the code runs.

*To write:* set `stage = "a2"`; the AC master (M^ac) solves once on the DC cut pool; compare `result.final` with `result.ac`; Knitro or Ipopt via `ac_solver`.

```python
from ropf import AlgorithmConfig, SolverConfig, read_matpower, run

net = read_matpower("data/case_ACTIVSg200.m")
result = run(net, AlgorithmConfig(metric="joule_loss_max", stage="a2"),
             ac_solver=SolverConfig(name="ipopt"))
result.final, result.ac
```

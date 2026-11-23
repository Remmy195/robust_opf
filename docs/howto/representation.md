# Choose an exposure representation

Set `metric` in the config or in `AlgorithmConfig`:

| `metric` | Representation | Cut family |
|---|---|---|
| `max_active_flow` | $\phi^{\mathrm{flow}}$ | two linear inequalities per line |
| `joule_loss_max` | $\phi^{\mathrm{joule}}$ | one convex quadratic per line |
| `bus_flow_sum_agg` | $\phi^{\mathrm{bus}}$ | one linear inequality per bus and sign pattern |

`flow_domain = rated` drops lines with no rating from the representation.

*To write:* when to pick each. See [Exposure](../explanation/exposure.md).

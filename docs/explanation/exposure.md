# Exposure

!!! note "Draft"
    Equations are in place. Prose marked *To write* and code outputs are open.

## Exposure: what and why?

*To write:* risk exposure defined once; disfigurements, the component-removal sets that mix buses and lines; why the surrogate targets the components an event class would hit hardest.

## Exposure representations

Each representation assigns a contribution to every component and takes the mean of the $q$ largest, $\operatorname{avg}_q$, padding with zeros when there are fewer than $q$ components.

| Representation | Contribution | Config `metric` |
|---|---|---|
| $\phi^{\mathrm{bus}}$ | $\sum_{(m,n) \in \mathcal{E}_i} \lvert P_{mn} \rvert$ at bus $i$ | `bus_flow_sum_agg` |
| $\phi^{\mathrm{flow}}$ | $\lvert P_{ij} \rvert$ on line $(i,j)$ | `max_active_flow` |
| $\phi^{\mathrm{joule}}$ | $R_{ij} P_{ij}^{2}$ on line $(i,j)$ | `joule_loss_max` |

$R_{ij}$ is the nonnegative part of the series resistance.

```python
from ropf import AlgorithmConfig, read_matpower, risk, run

net = read_matpower("data/case_ACTIVSg200.m")
nominal = run(net, AlgorithmConfig(weight_multiplier=0))
for metric in ("bus_flow_sum_agg", "max_active_flow", "joule_loss_max"):
    print(metric, risk.evaluate(metric, net, nominal.dispatch.Pf).value)
```

## Order-$q$ aggregation

At $q = 1$ each representation is the largest single contribution. The mean of the $q$ largest is the $q$-centrum objective of location theory, and for $q > 1$ it is the conditional value-at-risk of the contributions at tail fraction $q/|\mathcal{E}|$ (or $q/|\mathcal{N}|$) under uniform weights.

```python
from ropf.risk import top_q_mean
top_q_mean([5.0, 3.0, 2.0, 1.0], 2)
```

*To write:* output, and a short check that `top_q_mean` equals the max at $q = 1$ and the mean when $q$ counts every component.

## Exposure cuts

The master carries a threshold $t$ and an excess $u_c$ for each pooled component $c$:

$$
\begin{aligned}
& \Phi \ge t + \tfrac{1}{q} \textstyle\sum_{c \in \mathcal{P}_{k}} u_c \\
& u_{ij} \ge P_{ij} - t, \quad u_{ij} \ge -P_{ij} - t && \forall\, (i,j) \in \mathcal{P}_{k} && \text{(flow)} \\
& u_{ij} \ge R_{ij} P_{ij}^{2} - t && \forall\, (i,j) \in \mathcal{P}_{k} && \text{(Joule)} \\
& u_{i} \ge \textstyle\sum_{(m,n) \in \mathcal{E}_{i}} \sigma^{l}_{mn} P_{mn} - t && \forall\, i \in \mathcal{A}_{l},\ l \in \{1, \ldots, k\} && \text{(bus)} \\
& t \ge 0, \quad u_c \ge 0 && \forall\, c \in \mathcal{P}_{k}
\end{aligned}
$$

The right side of the first inequality is the mean of the $q$ largest pooled contributions. At $q = 1$ the families reduce to $\Phi \ge (\cdot)$ on each pooled component.

### The bus family and sign patterns

*To write:* $\sigma^{l}_{mn}$ is the sign the flow carried at the iteration $l$ that selected the bus; a bus selected under a new sign pattern gains a new inequality.

## Implementation

*To write:* run all three representations on ACTIVSg200 at $\lambda^{\mathrm{ref}}$ and print the exit row of each.

# Introductory theory

This page builds Algorithm 1 from Kelley's cutting-plane method. Each section ends with code that runs on ACTIVSg200.

!!! note "Draft"
    Section headings and equations are in place. Prose marked *To write* and every code output are open. Run the code locally with an AMPL licence and paste the output; the online build has no solver.

## Preliminaries: the DC master problem

Model (M) is a DC optimal power flow whose objective adds the exposure surrogate $\Phi$, priced at $\lambda$, to the generation cost:

$$
\begin{aligned}
(\mathrm{M}) \quad \min \quad & \sum_{i \in \mathcal{G}} \sum_{g \in \mathcal{G}_i} \bigl( C^{0}_g + C^{1}_g P_g + C^{2}_g P_g^{2} \bigr) + \lambda \Phi \\
\text{s.t.} \quad & P_{ij} = B_{ij} \theta_{ij} && \forall\, (i,j) \in \mathcal{E} \\
& \textstyle\sum_{g \in \mathcal{G}_i} P_g - P_{di} = \sum_{(i,j) \in \mathcal{E}} P_{ij} - \sum_{(j,i) \in \mathcal{E}} P_{ji} && \forall\, i \in \mathcal{N} \\
& -S^{\max}_{ij} \le P_{ij} \le S^{\max}_{ij} && \forall\, (i,j) \in \mathcal{E} \\
& P^{\min}_g \le P_g \le P^{\max}_g && \forall\, g \in \mathcal{G}_i,\ i \in \mathcal{G} \\
& \theta_{i_0} = 0, \quad \Phi \ge 0
\end{aligned}
$$

At $\lambda = 0$ the model returns the nominal dispatch.

```python
from ropf import AlgorithmConfig, read_matpower, run

net = read_matpower("data/case_ACTIVSg200.m")
nominal = run(net, AlgorithmConfig(weight_multiplier=0))
nominal.z0, nominal.rho0
```

The code names $\hat{\phi}$ `rho`. *To write:* the printed $z^{0}$ and $\hat{\phi}^{0}$.

## Preliminaries: Kelley's cutting-plane method

*To write:* a convex function as the maximum of its supporting hyperplanes; the lower model; one-dimensional example with bounds printed per iteration.

### Bounds

*To write:* $\Phi^{k} \le \hat{\phi}^{k}$ at every iteration; the relative gap $\Gamma^{k} = (\hat{\phi}^{k} - \Phi^{k})/\hat{\phi}^{k}$.

## Approximating the exposure term

*To write:* $\Phi$ starts unconstrained apart from $\Phi \ge 0$, so the first solve is the nominal dispatch; each cut family raises the lower model where the incumbent is most exposed.

## Implementation: component selection

At iteration $k$ component selection evaluates the representation at the incumbent and returns $\mathcal{A}_{k}$, the $\kappa$ components of largest contribution not yet cut. The pool is $\mathcal{P}_{k} = \mathcal{A}_{1} \cup \cdots \cup \mathcal{A}_{k}$.

```python
from ropf import risk

ev = risk.evaluate("joule_loss_max", net, nominal.dispatch.Pf)
risk.select(ev, kappa=5)
```

*To write:* show `select` on the nominal flows.

## Implementation: the master solve

*To write:* append the cut family for $\mathcal{A}_{k}$, solve (M), read back $\bm{x}^{k}$, $z^{k}$, $\Phi^{k}$.

## Implementation: the loop

1. Solve (M) with no exposure cuts; obtain $\bm{x}^{0}$, $z^{0}$, $\hat{\phi}^{0}$.
2. Repeat: select $\mathcal{A}_{k}$, append its cuts, solve (M), compute $\hat{\phi}^{k}$ and $\Gamma^{k}$.
3. Stop when $\hat{\phi}^{k}/\hat{\phi}^{0} \le 1 - \eta$, $\Gamma^{k} \le \delta$, or $k = k^{\max}$.

`algorithm.py` tags each step `ALG1-Ln` after the line of Algorithm 1 it implements.

```python
result = run(net, AlgorithmConfig(metric="joule_loss_max"))
for it in result.iterations:
    print(it.k, it.cost, it.surrogate, it.rho, it.gamma)
result.termination
```

*To write:* the printed iteration table.

## Implementation: the exposure weight

The weight is set relative to $\lambda^{\mathrm{ref}} = z^{0}/\hat{\phi}^{0}$, which equalizes the two objective terms at the nominal dispatch. `weight_multiplier` scales $\lambda^{\mathrm{ref}}$; `risk_weight` sets $\lambda$ directly.

## Implementation: the AC stage

*To write:* the AC master (M^ac) solved once on the DC cut pool (`stage = a2`); link to the [AC stage tutorial](../tutorials/ac_stage.md).

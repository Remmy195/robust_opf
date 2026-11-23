# Choose the order q

`q = 1` (the default) prices the largest single contribution. `q > 1` prices the mean of the $q$ largest.

```python
AlgorithmConfig(metric="joule_loss_max", q=10)
```

*To write:* guidance on $q$ relative to $\kappa$ and system size. See [Order-q aggregation](../explanation/exposure.md#order-q-aggregation).

# Example configs

Small `ropf solve` quickstarts on ACTIVSg200. The real campaign is
[`../ladder.conf`](../ladder.conf).

| File | Scenario | Stage | `weight_grid` |
|---|---|---|---|
| [dcopf_single.conf](dcopf_single.conf) | Plain DC OPF | `baseline` | `0` |
| [acopf_single.conf](acopf_single.conf) | Plain AC OPF | `a3` | `0` |
| [risk_aware_single.conf](risk_aware_single.conf) | Algorithm 1, one weight | `baseline` | `1` |
| [frontier_sweep.conf](frontier_sweep.conf) | 3-point sweep | `baseline` | `0, 1, 2` |

```bash
ropf solve configs/examples/<file> --dry-run
ropf solve configs/examples/<file>
```

`ropf keys` lists every key; unknown or repeated keys error out.

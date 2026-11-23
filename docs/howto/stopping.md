# Set the stopping rule

Algorithm 1 stops at the first of three tests:

| Key | Default | Test |
|---|---|---|
| `eta` | `0.85` | $\hat{\phi}^{k}/\hat{\phi}^{0} \le 1 - \eta$ |
| `gamma_tol` | `1e-6` | $\Gamma^{k} \le$ `gamma_tol` |
| `k_bar` | `40` | $k = k^{\max}$ |

Set `gamma_tol` explicitly in every config. `result.termination` names the test that fired.

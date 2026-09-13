# Paired analysis: does the GNN actually beat the baseline?

Every model in a given row shares its seed with the baseline it is compared against, so the differences below are paired. Marginal means with +/- 0.015 error bars would hide an effect of +0.005; pairing does not.

## regime A default

| Metric | baseline | proposed | per-seed delta | mean delta | seeds won | oracle headroom | share recovered |
|---|---|---|---|---|---|---|---|
| auc_roc | 0.6693 | 0.6690 | -0.0012, +0.0005, -0.0004 | -0.0004 ± 0.0009 | 1/3 | +0.0100 | -4% |
| auc_pr | 0.5354 | 0.5360 | +0.0007, +0.0004, +0.0006 | +0.0006 ± 0.0002 | 3/3 | +0.0166 | 4% |

## regime B structural

| Metric | baseline | proposed | per-seed delta | mean delta | seeds won | oracle headroom | share recovered |
|---|---|---|---|---|---|---|---|
| auc_roc | 0.6911 | 0.6967 | +0.0046, +0.0053, +0.0070 | +0.0056 ± 0.0012 | 3/3 | +0.0319 | 18% |
| auc_pr | 0.5895 | 0.5954 | +0.0055, +0.0044, +0.0078 | +0.0059 ± 0.0017 | 3/3 | +0.0383 | 15% |


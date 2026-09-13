# Paired analysis: does the GNN actually beat the baseline?

Every model in a given row shares its seed with the baseline it is compared against, so the differences below are paired. Marginal means with +/- 0.015 error bars would hide an effect of +0.005; pairing does not.

## regime A default

| Metric | baseline | proposed | per-seed delta | mean delta | seeds won | oracle headroom | share recovered |
|---|---|---|---|---|---|---|---|
| auc_roc | 0.6693 | 0.6699 | -0.0017, +0.0046, -0.0013 | +0.0005 ± 0.0035 | 1/3 | +0.0100 | 5% |
| auc_pr | 0.5354 | 0.5384 | +0.0007, +0.0092, -0.0009 | +0.0030 ± 0.0055 | 2/3 | +0.0166 | 18% |

## regime B structural

| Metric | baseline | proposed | per-seed delta | mean delta | seeds won | oracle headroom | share recovered |
|---|---|---|---|---|---|---|---|
| auc_roc | 0.6911 | 0.6963 | +0.0060, +0.0032, +0.0062 | +0.0052 ± 0.0017 | 3/3 | +0.0319 | 16% |
| auc_pr | 0.5895 | 0.5934 | +0.0063, +0.0020, +0.0034 | +0.0039 ± 0.0022 | 3/3 | +0.0383 | 10% |

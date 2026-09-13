# Results

Mean ± sd over up to 3 seeds. Each seed regenerates the graph, the cascade and the split, so the spread covers data variance as well as initialisation variance.

Dataset per seed: `{'n_instances': 12000, 'pos_rate': 0.36575, 'n_nodes': 8000, 'ego_size': 50, 'n_train': 7217, 'n_val': 1792, 'n_test': 2991}`

## Regime A (default) - baselines vs proposed model

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (handcrafted only) | 0.6693 ± 0.0071 | 0.5354 ± 0.0032 | 0.5597 ± 0.0081 | 0.4516 | 0.7497 |
| DeepWalk + MLP | 0.5390 ± 0.0105 | 0.3979 ± 0.0117 | 0.5313 ± 0.0091 | 0.3699 | 0.9452 |
| node2vec + MLP | 0.5456 ± 0.0122 | 0.4024 ± 0.0099 | 0.5341 ± 0.0050 | 0.3650 | 0.9957 |
| Plain GCN (no handcrafted fusion) | 0.6200 ± 0.0141 | 0.4884 ± 0.0114 | 0.5333 ± 0.0111 | 0.4100 | 0.7731 |
| Plain GAT (no handcrafted fusion) | 0.6605 ± 0.0128 | 0.5269 ± 0.0058 | 0.5541 ± 0.0093 | 0.4514 | 0.7177 |
| Full model: GCN + handcrafted | 0.6693 ± 0.0092 | 0.5373 ± 0.0041 | 0.5593 ± 0.0059 | 0.4513 | 0.7352 |
| Full model: GAT + handcrafted  (proposed) | 0.6699 ± 0.0101 | 0.5384 ± 0.0031 | 0.5515 ± 0.0090 | 0.4634 | 0.6846 |
| [diagnostic] LR + 2-hop active count | 0.6692 ± 0.0074 | 0.5351 ± 0.0029 | 0.5603 ± 0.0089 | 0.4462 | 0.7605 |
| [diagnostic] LR + true structural diversity | 0.6793 ± 0.0032 | 0.5521 ± 0.0058 | 0.5638 ± 0.0086 | 0.4733 | 0.6998 |

## Ablations (applied to the full GAT model, regime A)

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| ABL: - instance normalisation | 0.6603 ± 0.0000 | 0.5388 ± 0.0000 | 0.5630 ± 0.0000 | 0.4250 | 0.8335 |
| ABL: - wide/deep skip on handcrafted | 0.6582 ± 0.0000 | 0.5403 ± 0.0000 | 0.5587 ± 0.0000 | 0.4303 | 0.7964 |
| ABL: - handcrafted features entirely | 0.6474 ± 0.0000 | 0.5204 ± 0.0000 | 0.5528 ± 0.0000 | 0.4461 | 0.7267 |
| ABL: + node2vec structural embeddings | 0.6556 ± 0.0000 | 0.5342 ± 0.0000 | 0.5607 ± 0.0000 | 0.4298 | 0.8063 |
| ABL: 1 GNN layer instead of 2 | 0.6593 ± 0.0000 | 0.5376 ± 0.0000 | 0.5529 ± 0.0000 | 0.4342 | 0.7611 |
| ABL: 3 GNN layers instead of 2 | 0.6590 ± 0.0000 | 0.5406 ± 0.0000 | 0.5560 ± 0.0000 | 0.4347 | 0.7710 |

## Sensitivity to ego network radius (regime A)

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| KHOP: ego network radius k=1 | 0.6610 ± 0.0000 | 0.5409 ± 0.0000 | 0.5630 ± 0.0000 | 0.4340 | 0.8009 |
| KHOP: ego network radius k=2 | 0.6598 ± 0.0000 | 0.5389 ± 0.0000 | 0.5552 ± 0.0000 | 0.4419 | 0.7466 |
| KHOP: ego network radius k=3 | 0.6596 ± 0.0000 | 0.5390 ± 0.0000 | 0.5562 ± 0.0000 | 0.4435 | 0.7457 |

## Regime B (structure-dominant) - baselines vs proposed model

Same pipeline, same code, a cascade where the structural term carries most of the signal and the raw active-neighbour count carries almost none.

Dataset per seed: `{'n_instances': 12000, 'pos_rate': 0.38858333333333334, 'n_nodes': 8000, 'ego_size': 50, 'n_train': 7233, 'n_val': 1788, 'n_test': 2979}`

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (handcrafted only) | 0.6911 ± 0.0150 | 0.5895 ± 0.0052 | 0.5966 ± 0.0135 | 0.4924 | 0.7636 |
| DeepWalk + MLP | 0.5459 ± 0.0054 | 0.4254 ± 0.0173 | 0.5630 ± 0.0138 | 0.3938 | 0.9878 |
| node2vec + MLP | 0.5477 ± 0.0112 | 0.4284 ± 0.0058 | 0.5617 ± 0.0143 | 0.3925 | 0.9886 |
| Plain GCN (no handcrafted fusion) | 0.6483 ± 0.0187 | 0.5442 ± 0.0058 | 0.5740 ± 0.0111 | 0.4442 | 0.8286 |
| Plain GAT (no handcrafted fusion) | 0.6851 ± 0.0175 | 0.5805 ± 0.0057 | 0.5964 ± 0.0045 | 0.4763 | 0.7979 |
| Full model: GCN + handcrafted | 0.6968 ± 0.0124 | 0.5954 ± 0.0031 | 0.6057 ± 0.0081 | 0.4726 | 0.8431 |
| Full model: GAT + handcrafted  (proposed) | 0.6963 ± 0.0134 | 0.5934 ± 0.0038 | 0.6079 ± 0.0086 | 0.4917 | 0.7991 |
| [diagnostic] LR + 2-hop active count | 0.6911 ± 0.0152 | 0.5898 ± 0.0055 | 0.5955 ± 0.0177 | 0.4896 | 0.7676 |
| [diagnostic] LR + true structural diversity | 0.7230 ± 0.0112 | 0.6278 ± 0.0020 | 0.6199 ± 0.0080 | 0.5076 | 0.7981 |

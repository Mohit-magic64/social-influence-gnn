# Results

Mean ± sd over up to 3 seeds. Each seed regenerates the graph, the cascade and the split, so the spread covers data variance as well as initialisation variance.

Dataset per seed: `{'n_instances': 12000, 'pos_rate': 0.36575, 'n_nodes': 8000, 'ego_size': 50, 'n_train': 7217, 'n_val': 1792, 'n_test': 2991}`

## Regime A (default) - baselines vs proposed model

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (handcrafted only) | 0.6693 ± 0.0071 | 0.5354 ± 0.0032 | 0.5597 ± 0.0081 | 0.4516 | 0.7497 |
| DeepWalk + MLP | 0.5390 ± 0.0105 | 0.3979 ± 0.0117 | 0.5313 ± 0.0091 | 0.3699 | 0.9452 |
| node2vec + MLP | 0.5456 ± 0.0122 | 0.4024 ± 0.0099 | 0.5341 ± 0.0050 | 0.3650 | 0.9957 |
| Plain GCN (no handcrafted fusion) | 0.6234 ± 0.0105 | 0.4895 ± 0.0099 | 0.5386 ± 0.0098 | 0.4121 | 0.7825 |
| Plain GAT (no handcrafted fusion) | 0.6599 ± 0.0122 | 0.5257 ± 0.0059 | 0.5556 ± 0.0086 | 0.4442 | 0.7466 |
| Full model: GCN + handcrafted | 0.6703 ± 0.0112 | 0.5374 ± 0.0013 | 0.5579 ± 0.0116 | 0.4457 | 0.7484 |
| Full model: GAT + handcrafted  (proposed) | 0.6690 ± 0.0079 | 0.5360 ± 0.0033 | 0.5545 ± 0.0115 | 0.4551 | 0.7240 |
| [diagnostic] LR + 2-hop active count | 0.6692 ± 0.0074 | 0.5351 ± 0.0029 | 0.5603 ± 0.0089 | 0.4462 | 0.7605 |
| [diagnostic] LR + true structural diversity | 0.6793 ± 0.0032 | 0.5521 ± 0.0058 | 0.5638 ± 0.0086 | 0.4733 | 0.6998 |

## Ablations (applied to the full GAT model, regime A)

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| ABL: - instance normalisation | 0.6592 ± 0.0000 | 0.5351 ± 0.0000 | 0.5613 ± 0.0000 | 0.4366 | 0.7855 |
| ABL: - wide/deep skip on handcrafted | 0.6593 ± 0.0000 | 0.5399 ± 0.0000 | 0.5544 ± 0.0000 | 0.4431 | 0.7403 |
| ABL: - handcrafted features entirely | 0.6437 ± 0.0000 | 0.5105 ± 0.0000 | 0.5509 ± 0.0000 | 0.4443 | 0.7249 |
| ABL: + node2vec structural embeddings | 0.6564 ± 0.0000 | 0.5316 ± 0.0000 | 0.5625 ± 0.0000 | 0.4107 | 0.8923 |
| ABL: 1 GNN layer instead of 2 | 0.6597 ± 0.0000 | 0.5410 ± 0.0000 | 0.5518 ± 0.0000 | 0.4343 | 0.7566 |
| ABL: 3 GNN layers instead of 2 | 0.6514 ± 0.0000 | 0.5276 ± 0.0000 | 0.5558 ± 0.0000 | 0.4289 | 0.7891 |

## Sensitivity to ego network radius (regime A)

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| KHOP: ego network radius k=1 | 0.6586 ± 0.0000 | 0.5342 ± 0.0000 | 0.5560 ± 0.0000 | 0.4351 | 0.7701 |
| KHOP: ego network radius k=2 | 0.6554 ± 0.0000 | 0.5328 ± 0.0000 | 0.5543 ± 0.0000 | 0.4218 | 0.8081 |
| KHOP: ego network radius k=3 | 0.6624 ± 0.0000 | 0.5413 ± 0.0000 | 0.5563 ± 0.0000 | 0.4427 | 0.7484 |

## Regime B (structure-dominant) - baselines vs proposed model

Same pipeline, same code, a cascade where the structural term carries most of the signal and the raw active-neighbour count carries almost none.

Dataset per seed: `{'n_instances': 12000, 'pos_rate': 0.38858333333333334, 'n_nodes': 8000, 'ego_size': 50, 'n_train': 7233, 'n_val': 1788, 'n_test': 2979}`

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (handcrafted only) | 0.6911 ± 0.0150 | 0.5895 ± 0.0052 | 0.5966 ± 0.0135 | 0.4924 | 0.7636 |
| DeepWalk + MLP | 0.5459 ± 0.0054 | 0.4254 ± 0.0173 | 0.5630 ± 0.0138 | 0.3938 | 0.9878 |
| node2vec + MLP | 0.5477 ± 0.0112 | 0.4284 ± 0.0058 | 0.5617 ± 0.0143 | 0.3925 | 0.9886 |
| Plain GCN (no handcrafted fusion) | 0.6504 ± 0.0177 | 0.5482 ± 0.0025 | 0.5740 ± 0.0123 | 0.4475 | 0.8232 |
| Plain GAT (no handcrafted fusion) | 0.6856 ± 0.0151 | 0.5823 ± 0.0054 | 0.5943 ± 0.0043 | 0.4792 | 0.7824 |
| Full model: GCN + handcrafted | 0.6929 ± 0.0116 | 0.5919 ± 0.0021 | 0.6014 ± 0.0026 | 0.4704 | 0.8366 |
| Full model: GAT + handcrafted  (proposed) | 0.6967 ± 0.0145 | 0.5954 ± 0.0041 | 0.6071 ± 0.0052 | 0.4829 | 0.8188 |
| [diagnostic] LR + 2-hop active count | 0.6911 ± 0.0152 | 0.5898 ± 0.0055 | 0.5955 ± 0.0177 | 0.4896 | 0.7676 |
| [diagnostic] LR + true structural diversity | 0.7230 ± 0.0112 | 0.6278 ± 0.0020 | 0.6199 ± 0.0080 | 0.5076 | 0.7981 |


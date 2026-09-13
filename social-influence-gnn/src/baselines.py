"""
The non-GNN comparison points from section 4 of the problem statement.

Both baselines get every advantage that costs nothing: standardised inputs, the same
split, class weighting, a hyperparameter sweep for LR, and an aggregated neighbourhood
view for the embedding model. If the GNN wins it should win on information access, not
because someone else was tuned badly.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics, pick_threshold


def _fit_eval(clf, Xtr, ytr, Xva, yva, Xte, yte):
    """Fit, pick the threshold on val, score test once."""
    clf.fit(Xtr, ytr)
    sv = clf.predict_proba(Xva)[:, 1]
    thr = pick_threshold(yva, sv)
    st = clf.predict_proba(Xte)[:, 1]
    return compute_metrics(yte, st, thr), st
    # Same val-then-test discipline the GNN gets. Baselines evaluated more loosely than
    # the proposed model is the most common way these comparisons get quietly rigged.


def logistic_regression_baseline(inst, train_idx, val_idx, test_idx, seed=0,
                                 extra: np.ndarray | None = None,
                                 include_2hop: bool = False):
    """Handcrafted features only. The bar the GNN has to clear."""
    from data import N_PS_FEATURES
    X = inst.hand if include_2hop else inst.hand[:, :N_PS_FEATURES]
    # Default excludes the 2-hop count. It is not in the PS feature list and it is a
    # hand-built proxy for what the GNN should learn, so it gets its own reported row
    # rather than being folded in silently.
    if extra is not None:
        X = np.concatenate([X, extra], axis=1)
    sc = StandardScaler().fit(X[train_idx])
    Xs = sc.transform(X)
    # Fit the scaler on train only, same rule as the GNN's feature scaler.
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        {"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        scoring="average_precision", cv=3, n_jobs=-1,
    )
    # Sweep the regularisation strength and score on average precision, so the baseline
    # is tuned for the same metric the GNN is early-stopped on. class_weight="balanced"
    # mirrors the pos_weight the GNN gets.
    return _fit_eval(grid, Xs[train_idx], inst.y[train_idx], Xs[val_idx],
                     inst.y[val_idx], Xs[test_idx], inst.y[test_idx])


def embedding_mlp_baseline(inst, emb: np.ndarray, train_idx, val_idx, test_idx, seed=0):
    """Unsupervised walk embeddings into an MLP. No message passing at predict time."""
    ids, act, valid = inst.node_ids, inst.active.astype(bool), inst.valid.astype(bool)
    M, s = ids.shape
    d = emb.shape[1]

    ego = emb[ids[:, 0]]
    E = emb[ids]                                        # [M, s, d]
    nbr_mask = valid.copy(); nbr_mask[:, 0] = False
    # Exclude the ego from the neighbour aggregates, or its own vector gets counted twice.
    act_mask = act & nbr_mask

    def masked_mean(mask):
        w = mask[..., None].astype(np.float32)
        return (E * w).sum(1) / np.clip(w.sum(1), 1e-6, None)
        # clip handles ego networks with zero active neighbours, which would be 0/0.

    n_act = act_mask.sum(1, keepdims=True).astype(np.float32)
    n_nbr = np.clip(nbr_mask.sum(1, keepdims=True).astype(np.float32), 1, None)
    X = np.concatenate([ego, masked_mean(act_mask), masked_mean(nbr_mask),
                        np.log1p(n_act), n_act / n_nbr], axis=1).astype(np.float32)
    # The last two columns are handed over deliberately. Without any action signal this
    # baseline would be predicting adoption from graph position alone, and it would lose
    # for the wrong reason. This is a strengthened version of the PS baseline.

    sc = StandardScaler().fit(X[train_idx])
    Xs = sc.transform(X)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400,
                        early_stopping=True, n_iter_no_change=15, random_state=seed)
    return _fit_eval(clf, Xs[train_idx], inst.y[train_idx], Xs[val_idx],
                     inst.y[val_idx], Xs[test_idx], inst.y[test_idx])


def structural_diversity_oracle(inst, sg, snapshots: dict, train_idx, val_idx, test_idx,
                                seed=0):
    """Diagnostic: logistic regression handed the exact generative feature."""
    # Not a competitor. It is a ceiling. It bounds how much any model could gain from
    # structure, which turns "is the GNN better" into "what share of the available
    # headroom did it capture". Without it, a +0.005 gain is uninterpretable.
    from data import _active_component_count
    adj_sets = [set(a.tolist()) for a in sg.adj_list]
    div = np.zeros((len(inst.y), 1), dtype=np.float32)
    for i, v in enumerate(inst.ego):
        active = snapshots[int(inst.round_idx[i])]
        # Look up the snapshot this row was actually cut at. Using one shared snapshot
        # was a real bug here: it corrupted two thirds of the rows and made structural
        # diversity look worthless.
        nbrs = sg.adj_list[v]
        act_nbrs = nbrs[active[nbrs]]
        div[i, 0] = np.log1p(_active_component_count(act_nbrs, adj_sets))
        # log1p to match the transform the cascade itself applies.
    return logistic_regression_baseline(inst, train_idx, val_idx, test_idx, seed,
                                        extra=div)

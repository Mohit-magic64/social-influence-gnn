"""
baselines.py
------------
The three non-GNN comparison points from section 4 of the problem statement.

A note on fairness, because this is where projects like this usually cheat: it is easy
to make a proposed model look good by handicapping the baselines. Both baselines here
are given every advantage that costs nothing - standardised inputs, the same
train/val/test split, class weighting, a hyperparameter sweep for logistic regression,
and for the embedding baseline an aggregated view of the neighbourhood rather than just
the ego's own vector. If the GNN wins, it should win on information access, not on
someone else being tuned badly.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics, pick_threshold


def _fit_eval(clf, Xtr, ytr, Xva, yva, Xte, yte):
    clf.fit(Xtr, ytr)
    sv = clf.predict_proba(Xva)[:, 1]
    thr = pick_threshold(yva, sv)
    st = clf.predict_proba(Xte)[:, 1]
    return compute_metrics(yte, st, thr), st


def logistic_regression_baseline(inst, train_idx, val_idx, test_idx, seed=0,
                                 extra: np.ndarray | None = None,
                                 include_2hop: bool = False):
    """Handcrafted features only, no graph embedding. The 'does the GNN earn its keep?' bar."""
    from data import N_PS_FEATURES
    X = inst.hand if include_2hop else inst.hand[:, :N_PS_FEATURES]
    if extra is not None:
        X = np.concatenate([X, extra], axis=1)
    sc = StandardScaler().fit(X[train_idx])
    Xs = sc.transform(X)
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        {"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        scoring="average_precision", cv=3, n_jobs=-1,
    )
    return _fit_eval(grid, Xs[train_idx], inst.y[train_idx], Xs[val_idx],
                     inst.y[val_idx], Xs[test_idx], inst.y[test_idx])


def embedding_mlp_baseline(inst, emb: np.ndarray, train_idx, val_idx, test_idx, seed=0):
    """
    Unsupervised walk embeddings -> MLP. No message passing at prediction time; the
    graph only enters through pretrained vectors.

    The feature vector is [ego emb || mean emb of ACTIVE neighbours || mean emb of ALL
    neighbours || #active || fraction active]. The last two are handed over on purpose:
    without any action signal this baseline is predicting adoption from position in the
    graph alone, which is not a fair comparison - it would lose for the wrong reason.
    """
    ids, act, valid = inst.node_ids, inst.active.astype(bool), inst.valid.astype(bool)
    M, s = ids.shape
    d = emb.shape[1]

    ego = emb[ids[:, 0]]
    E = emb[ids]                                        # [M, s, d]
    nbr_mask = valid.copy(); nbr_mask[:, 0] = False
    act_mask = act & nbr_mask

    def masked_mean(mask):
        w = mask[..., None].astype(np.float32)
        return (E * w).sum(1) / np.clip(w.sum(1), 1e-6, None)

    n_act = act_mask.sum(1, keepdims=True).astype(np.float32)
    n_nbr = np.clip(nbr_mask.sum(1, keepdims=True).astype(np.float32), 1, None)
    X = np.concatenate([ego, masked_mean(act_mask), masked_mean(nbr_mask),
                        np.log1p(n_act), n_act / n_nbr], axis=1).astype(np.float32)

    sc = StandardScaler().fit(X[train_idx])
    Xs = sc.transform(X)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400,
                        early_stopping=True, n_iter_no_change=15, random_state=seed)
    return _fit_eval(clf, Xs[train_idx], inst.y[train_idx], Xs[val_idx],
                     inst.y[val_idx], Xs[test_idx], inst.y[test_idx])


def structural_diversity_oracle(inst, sg, snapshots: dict, train_idx, val_idx, test_idx,
                                seed=0):
    """
    Diagnostic, not a headline baseline: logistic regression PLUS the exact structural
    -diversity term the simulator uses. It is the ceiling for a feature-engineered model
    that already knows the answer. If the GNN lands near this, the GNN has recovered the
    mechanism from raw wiring rather than been told it.
    """
    from data import _active_component_count
    adj_sets = [set(a.tolist()) for a in sg.adj_list]
    div = np.zeros((len(inst.y), 1), dtype=np.float32)
    for i, v in enumerate(inst.ego):
        # each instance was cut at its own observation round - using one shared
        # snapshot silently corrupts two thirds of the rows
        active = snapshots[int(inst.round_idx[i])]
        nbrs = sg.adj_list[v]
        act_nbrs = nbrs[active[nbrs]]
        div[i, 0] = np.log1p(_active_component_count(act_nbrs, adj_sets))
    return logistic_regression_baseline(inst, train_idx, val_idx, test_idx, seed,
                                        extra=div)

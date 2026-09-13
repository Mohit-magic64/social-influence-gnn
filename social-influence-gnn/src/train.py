"""
train.py
--------
Batching and the training loop. BCE with logits, Adam, early stopping on validation
AUC-PR, best-checkpoint restore.

Node features fed to the GNN are assembled here rather than stored on disk, so that
ablations can switch channels on and off without rebuilding the dataset.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from evaluate import compute_metrics, pick_threshold


class InstanceBatcher:
    """
    Assembles [B, s, F] node-feature tensors on the fly.

    Node feature channels:
        0  action state of this node at observation time (ego forced to 0)
        1  is-ego indicator
        2  log degree (global)
        3  clustering coefficient (global)
        4  attr: log activity
        5  attr: tenure
        6  attr: historical action rate
      (+) optional structural embedding block, concatenated inside the model

    Channel 0 is the whole task. Channel 1 matters more than it looks: after two rounds
    of message passing the ego's representation is a blend of its neighbourhood, and
    without an explicit marker the model cannot tell which row is the person it is
    being asked about.
    """

    def __init__(self, inst, sg, struct_emb=None, device="cpu"):
        self.inst = inst
        self.device = device
        N = sg.n_nodes
        deg = np.array([sg.adj_list[i].size for i in range(N)], dtype=np.float32)
        import networkx as nx
        clus = np.zeros(N, dtype=np.float32)
        for node, c in nx.clustering(sg.G).items():
            clus[node] = c
        self.node_table = np.concatenate(
            [np.log1p(deg)[:, None], clus[:, None], sg.attrs], axis=1
        ).astype(np.float32)                              # [N, 5]
        self.struct_emb = struct_emb
        self.d_node = 2 + self.node_table.shape[1]
        self.d_struct = 0 if struct_emb is None else struct_emb.shape[1]

    def make(self, idx: np.ndarray):
        ins = self.inst
        ids = ins.node_ids[idx]
        mask = torch.tensor(ins.valid[idx], dtype=torch.float32, device=self.device)
        act = torch.tensor(ins.active[idx], dtype=torch.float32, device=self.device)

        tab = torch.tensor(self.node_table[ids], dtype=torch.float32, device=self.device)
        is_ego = torch.zeros_like(act)
        is_ego[:, 0] = 1.0
        x = torch.cat([act.unsqueeze(-1), is_ego.unsqueeze(-1), tab], dim=-1)
        x = x * mask.unsqueeze(-1)

        struct = None
        if self.struct_emb is not None:
            struct = torch.tensor(self.struct_emb[ids], dtype=torch.float32,
                                  device=self.device) * mask.unsqueeze(-1)

        adj = torch.tensor(ins.adj[idx], dtype=torch.float32, device=self.device)
        hand = torch.tensor(self.hand_scaled[idx], dtype=torch.float32, device=self.device)
        y = torch.tensor(ins.y[idx], dtype=torch.float32, device=self.device)
        return x, adj, mask, hand, struct, y

    def fit_hand_scaler(self, train_idx: np.ndarray):
        """Standardise handcrafted features using TRAIN statistics only."""
        from data import N_PS_FEATURES
        h = self.inst.hand[:, :N_PS_FEATURES]
        mu = h[train_idx].mean(0)
        sd = h[train_idx].std(0)
        sd[sd < 1e-6] = 1.0
        self.hand_scaled = ((h - mu) / sd).astype(np.float32)
        self.d_hand = self.hand_scaled.shape[1]
        return mu, sd


def train_model(model, batcher, train_idx, val_idx, test_idx, epochs=40, batch_size=128,
                lr=3e-3, weight_decay=5e-4, patience=8, seed=0, verbose=False,
                device="cpu"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)

    y_train = batcher.inst.y[train_idx]
    # Class weighting instead of resampling: it keeps every example in every epoch and
    # leaves the validation/test distributions untouched.
    pos_w = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=3)

    best_val, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            x, adj, mask, hand, struct, y = batcher.make(idx)
            logits = model(x, adj, mask, hand, struct)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1

        val_scores = predict(model, batcher, val_idx, batch_size, device)
        from sklearn.metrics import average_precision_score
        val_ap = average_precision_score(batcher.inst.y[val_idx], val_scores)
        sched.step(val_ap)
        if verbose:
            print(f"    ep{ep + 1:>3}  loss={tot / max(nb,1):.4f}  val_ap={val_ap:.4f}")

        if val_ap > best_val + 1e-5:
            best_val, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_scores = predict(model, batcher, val_idx, batch_size, device)
    thr = pick_threshold(batcher.inst.y[val_idx], val_scores)
    test_scores = predict(model, batcher, test_idx, batch_size, device)
    metrics = compute_metrics(batcher.inst.y[test_idx], test_scores, thr)
    return model, metrics, test_scores


@torch.no_grad()
def predict(model, batcher, idx, batch_size=256, device="cpu"):
    model.eval()
    out = []
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        x, adj, mask, hand, struct, _ = batcher.make(b)
        out.append(torch.sigmoid(model(x, adj, mask, hand, struct)).cpu().numpy())
    return np.concatenate(out)

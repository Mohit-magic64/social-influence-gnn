"""
Batching and the training loop.

Node features are assembled per batch rather than stored on disk, so ablations can turn
input channels on and off without rebuilding the dataset.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from evaluate import compute_metrics, pick_threshold


class InstanceBatcher:
    """Turns instance indices into the [B, s, F] tensors the model expects."""

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
        # One global lookup table of per-node features. Instances store node IDs only,
        # so this is gathered at batch time. Storing features per instance instead would
        # duplicate every node once per ego network it appears in.
        self.struct_emb = struct_emb
        self.d_node = 2 + self.node_table.shape[1]
        # The 2 is the action state and the is-ego flag, added in make() below.
        self.d_struct = 0 if struct_emb is None else struct_emb.shape[1]

    def make(self, idx: np.ndarray):
        """Build one batch. Node channels: [action_state, is_ego, deg, clustering, 3 attrs]."""
        ins = self.inst
        ids = ins.node_ids[idx]
        mask = torch.tensor(ins.valid[idx], dtype=torch.float32, device=self.device)
        act = torch.tensor(ins.active[idx], dtype=torch.float32, device=self.device)
        # act is the whole task: who around this user has already adopted.

        tab = torch.tensor(self.node_table[ids], dtype=torch.float32, device=self.device)
        is_ego = torch.zeros_like(act)
        is_ego[:, 0] = 1.0
        # Matters more than it looks. After two rounds of message passing the ego's
        # representation is a blend of its neighbourhood, so without an explicit marker
        # the model cannot tell which row is the person it is being asked about.
        x = torch.cat([act.unsqueeze(-1), is_ego.unsqueeze(-1), tab], dim=-1)
        x = x * mask.unsqueeze(-1)
        # Zero out padded rows so they contribute nothing to any aggregation.

        struct = None
        if self.struct_emb is not None:
            struct = torch.tensor(self.struct_emb[ids], dtype=torch.float32,
                                  device=self.device) * mask.unsqueeze(-1)

        adj = torch.tensor(ins.adj[idx], dtype=torch.float32, device=self.device)
        # uint8 on disk, float here. Casting per batch keeps the stored dataset at 30 MB.
        hand = torch.tensor(self.hand_scaled[idx], dtype=torch.float32, device=self.device)
        y = torch.tensor(ins.y[idx], dtype=torch.float32, device=self.device)
        return x, adj, mask, hand, struct, y

    def fit_hand_scaler(self, train_idx: np.ndarray):
        """Standardise handcrafted features using train statistics only."""
        from data import N_PS_FEATURES
        h = self.inst.hand[:, :N_PS_FEATURES]
        # Slice off the 2-hop feature. It is stored but excluded from the model, and is
        # only used by one clearly-labelled diagnostic baseline.
        mu = h[train_idx].mean(0)
        sd = h[train_idx].std(0)
        # Fit on train, apply everywhere. Fitting on the full array leaks test
        # distribution into training, which is subtle and does inflate scores.
        sd[sd < 1e-6] = 1.0
        # Constant column would divide by ~0 and produce inf.
        self.hand_scaled = ((h - mu) / sd).astype(np.float32)
        self.d_hand = self.hand_scaled.shape[1]
        return mu, sd


def train_model(model, batcher, train_idx, val_idx, test_idx, epochs=40, batch_size=128,
                lr=3e-3, weight_decay=5e-4, patience=8, seed=0, verbose=False,
                device="cpu"):
    """Train with class-weighted BCE, early stop on validation AUC-PR, score test once."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)

    y_train = batcher.inst.y[train_idx]
    pos_w = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    # Weight the positive class by the negative/positive ratio rather than resampling.
    # Weighting keeps every example in every epoch and leaves the val/test distributions
    # untouched, so the reported metrics still describe the real class balance.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=3)
    # mode="max" because we are tracking AUC-PR, where higher is better. Leaving it on
    # the default "min" would halve the learning rate exactly when things improve.

    best_val, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        # Reshuffle every epoch so batch composition changes.
        tot, nb = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            x, adj, mask, hand, struct, y = batcher.make(idx)
            logits = model(x, adj, mask, hand, struct)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # Cheap insurance. Attention softmax can produce large gradients early on.
            opt.step()
            tot += loss.item(); nb += 1

        val_scores = predict(model, batcher, val_idx, batch_size, device)
        from sklearn.metrics import average_precision_score
        val_ap = average_precision_score(batcher.inst.y[val_idx], val_scores)
        # Early stopping tracks AUC-PR, not loss. Loss is dominated by easy negatives
        # here, so it keeps falling while ranking quality has already peaked.
        sched.step(val_ap)
        if verbose:
            print(f"    ep{ep + 1:>3}  loss={tot / max(nb,1):.4f}  val_ap={val_ap:.4f}")

        if val_ap > best_val + 1e-5:
            best_val, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
            # deepcopy, not a reference. The live tensors keep training and a reference
            # would silently point at the final weights instead of the best ones.
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        # Restore the best epoch. Without this, early stopping returns a model that is
        # `patience` epochs past its peak, which on this data is visibly overfit.

    val_scores = predict(model, batcher, val_idx, batch_size, device)
    thr = pick_threshold(batcher.inst.y[val_idx], val_scores)
    test_scores = predict(model, batcher, test_idx, batch_size, device)
    metrics = compute_metrics(batcher.inst.y[test_idx], test_scores, thr)
    # Threshold chosen on validation, then frozen and applied to test. Tuning it on test
    # would make every F1 in the results table optimistic.
    return model, metrics, test_scores


@torch.no_grad()
def predict(model, batcher, idx, batch_size=256, device="cpu"):
    """Sigmoid probabilities for the given instance indices."""
    # no_grad saves memory and time; .eval() turns off dropout. Forgetting .eval() is a
    # classic quiet bug: predictions get noisier and scores drop for no visible reason.
    model.eval()
    out = []
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        x, adj, mask, hand, struct, _ = batcher.make(b)
        out.append(torch.sigmoid(model(x, adj, mask, hand, struct)).cpu().numpy())
    return np.concatenate(out)

"""
export_model.py
---------------
Trains the proposed model (GAT + handcrafted fusion + wide/deep skip) on one regime and
saves a self-contained checkpoint to weights/.

The checkpoint carries the architecture config and the feature scaler alongside the
state dict. A bare state_dict is not a deliverable - whoever loads it has to guess the
hidden size, the feature order and the normalisation constants, and a silently wrong
guess produces plausible garbage rather than an error.

    python src/export_model.py --regime default
    python src/export_model.py --regime structural
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import PS_HAND_FEATURES, build_dataset, split_indices
from models import InfluenceModel
from train import InstanceBatcher, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
WEIGHTS = os.path.join(ROOT, "weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="default", choices=["default", "structural"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-nodes", type=int, default=8000)
    ap.add_argument("--ego-size", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()

    os.makedirs(WEIGHTS, exist_ok=True)
    sg, _, inst, _ = build_dataset(cache_dir=CACHE, seed=args.seed,
                                   n_nodes=args.n_nodes, ego_size=args.ego_size,
                                   k_hops=2, regime=args.regime)
    tr, va, te = split_indices(inst, seed=args.seed)
    b = InstanceBatcher(inst, sg, struct_emb=None)
    mu, sd = b.fit_hand_scaler(tr)

    cfg = dict(d_node=b.d_node, d_hand=b.d_hand, hidden=64, layers=2, encoder="gat",
               heads=4, dropout=0.4, use_instance_norm=True, use_hand=True,
               use_struct_emb=False, d_struct=0, use_wide=True)
    model = InfluenceModel(**cfg)
    model, metrics, _ = train_model(model, b, tr, va, te, epochs=args.epochs,
                                    patience=5, lr=2e-3, seed=args.seed)

    path = os.path.join(WEIGHTS, f"influence_gat_{args.regime}_seed{args.seed}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg,
        "hand_feature_names": PS_HAND_FEATURES,
        "hand_scaler_mean": mu.tolist(),
        "hand_scaler_std": sd.tolist(),
        "node_feature_channels": ["active_state", "is_ego", "log_degree",
                                  "clustering_coef", "attr_log_activity",
                                  "attr_tenure", "attr_hist_rate"],
        "decision_threshold": metrics["threshold"],
        "test_metrics": metrics,
        "regime": args.regime,
        "seed": args.seed,
    }, path)
    print(f"saved {path}")
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()
                      if isinstance(v, float)}, indent=2))


if __name__ == "__main__":
    main()

"""
Runs every experiment and writes results/metrics.json and results/results.md.

    python src/experiments.py --stage all --seeds 0 1 2

Each stage checkpoints to results/partial/ and resumes, so a small machine can run this
in pieces without redoing finished models.

One rule holds throughout: the test set is touched exactly once per model, at the end,
with a threshold chosen on validation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (embedding_mlp_baseline, logistic_regression_baseline,
                       structural_diversity_oracle)
from data import build_dataset, split_indices
from embeddings import get_embeddings
from evaluate import aggregate, fmt_row
from models import InfluenceModel
from train import InstanceBatcher, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
RESULTS = os.path.join(ROOT, "results")
PARTIAL = os.path.join(RESULTS, "partial")

# Display order for the results table. Anything not listed here (ablations, k-hop rows)
# gets grouped separately in stage_report.
ORDER = [
    "Logistic Regression (handcrafted only)",
    "DeepWalk + MLP",
    "node2vec + MLP",
    "Plain GCN (no handcrafted fusion)",
    "Plain GAT (no handcrafted fusion)",
    "Full model: GCN + handcrafted",
    "Full model: GAT + handcrafted  (proposed)",
    "[diagnostic] LR + 2-hop active count",
    "[diagnostic] LR + true structural diversity",
]


def _key(stage, seed, regime):
    return f"{stage}_{regime}_seed{seed}"


def save_partial(stage, seed, payload, regime="default"):
    os.makedirs(PARTIAL, exist_ok=True)
    with open(os.path.join(PARTIAL, f"{_key(stage, seed, regime)}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def load_partial(stage, seed, regime="default"):
    """Resume support: re-running a stage skips models it already finished."""
    path = os.path.join(PARTIAL, f"{_key(stage, seed, regime)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def setup(seed, args, k_hops=2):
    sg, active_at, inst, snaps = build_dataset(
        cache_dir=CACHE, seed=seed, n_nodes=args.n_nodes,
        ego_size=args.ego_size, k_hops=k_hops, regime=args.regime)
    tr, va, te = split_indices(inst, seed=seed)
    return sg, active_at, inst, snaps, tr, va, te


def run_gnn(batcher, tr, va, te, seed, args, **kw):
    torch.manual_seed(seed)
    # Seed BEFORE constructing the model. The layers are randomly initialised at
    # construction time, so seeding only inside train_model left the init unseeded and
    # the same seed could swing the test AUC by ~0.007 between runs.
    model = InfluenceModel(
        d_node=batcher.d_node, d_hand=batcher.d_hand, hidden=64,
        layers=kw.get("layers", 2), encoder=kw.get("encoder", "gat"), heads=4,
        dropout=0.4, use_instance_norm=kw.get("use_inorm", True),
        use_hand=kw.get("use_hand", True), use_struct_emb=kw.get("use_struct", True),
        d_struct=batcher.d_struct, use_wide=kw.get("use_wide", True))
    t = time.time()
    _, m, _ = train_model(model, batcher, tr, va, te, epochs=args.epochs,
                          patience=args.patience, lr=2e-3, seed=seed,
                          verbose=args.verbose)
    m["train_seconds"] = round(time.time() - t, 1)
    # Every variant goes through this one function, so ablations cannot accidentally
    # differ from the main model in some hyperparameter nobody was tracking.
    return m


def make_batcher(inst, sg, seed, args, struct=False):
    """Batcher for this dataset. struct=True adds the node2vec channel (an ablation)."""
    # Default is False: the embeddings hurt the model, so they are opt-in rather than
    # part of the proposed architecture.
    emb = None
    if struct:
        emb = get_embeddings(sg.adj_list, sg.n_nodes, "node2vec", dim=64,
                             cache_dir=CACHE, seed=seed, verbose=args.verbose)
    return InstanceBatcher(inst, sg, struct_emb=emb)


# --------------------------------------------------------------------------------------

def stage_baselines(seed, args):
    sg, _, inst, snaps, tr, va, te = setup(seed, args)
    print(f"  instances={len(inst.y)} pos_rate={inst.y.mean():.3f} "
          f"split={len(tr)}/{len(va)}/{len(te)}", flush=True)
    out = {"_meta": {"n_instances": int(len(inst.y)), "pos_rate": float(inst.y.mean()),
                     "n_nodes": args.n_nodes, "ego_size": args.ego_size,
                     "n_train": len(tr), "n_val": len(va), "n_test": len(te)}}

    dw = get_embeddings(sg.adj_list, sg.n_nodes, "deepwalk", dim=64, cache_dir=CACHE,
                        seed=seed, verbose=args.verbose)
    n2v = get_embeddings(sg.adj_list, sg.n_nodes, "node2vec", dim=64, cache_dir=CACHE,
                         seed=seed, verbose=args.verbose)

    out["Logistic Regression (handcrafted only)"] = \
        logistic_regression_baseline(inst, tr, va, te, seed)[0]
    out["[diagnostic] LR + 2-hop active count"] = \
        logistic_regression_baseline(inst, tr, va, te, seed, include_2hop=True)[0]
    out["DeepWalk + MLP"] = embedding_mlp_baseline(inst, dw, tr, va, te, seed)[0]
    out["node2vec + MLP"] = embedding_mlp_baseline(inst, n2v, tr, va, te, seed)[0]

    out["[diagnostic] LR + true structural diversity"] = \
        structural_diversity_oracle(inst, sg, snaps, tr, va, te, seed)[0]

    for k, v in out.items():
        if k != "_meta":
            print(f"  {k:<46} AUC={v['auc_roc']:.4f} AP={v['auc_pr']:.4f}", flush=True)
    save_partial("baselines", seed, out, args.regime)
    return out


def stage_models(seed, args):
    sg, _, inst, _, tr, va, te = setup(seed, args)
    b = make_batcher(inst, sg, seed, args)
    b.fit_hand_scaler(tr)
    out = load_partial("models", seed, args.regime)
    for name, kw in [
        ("Plain GCN (no handcrafted fusion)", dict(encoder="gcn", use_hand=False)),
        ("Plain GAT (no handcrafted fusion)", dict(encoder="gat", use_hand=False)),
        ("Full model: GCN + handcrafted", dict(encoder="gcn", use_hand=True)),
        ("Full model: GAT + handcrafted  (proposed)", dict(encoder="gat", use_hand=True)),
    ]:
        if name in out:
            continue
        out[name] = run_gnn(b, tr, va, te, seed, args, **kw)
        print(f"  {name:<46} AUC={out[name]['auc_roc']:.4f} "
              f"AP={out[name]['auc_pr']:.4f} ({out[name]['train_seconds']}s)", flush=True)
        save_partial("models", seed, out, args.regime)
    return out


def stage_ablations(seed, args):
    sg, _, inst, _, tr, va, te = setup(seed, args)
    b = make_batcher(inst, sg, seed, args)
    b.fit_hand_scaler(tr)
    b_st = make_batcher(inst, sg, seed, args, struct=True)
    b_st.hand_scaled, b_st.d_hand = b.hand_scaled, b.d_hand
    # Reuse the same fitted scaler, so the only difference between the two batchers is
    # the extra embedding channel. Refitting would add a second confound.
    out = load_partial("ablations", seed, args.regime)
    for name, bb, kw in [
        ("ABL: - instance normalisation", b, dict(use_inorm=False)),
        ("ABL: - wide/deep skip on handcrafted", b, dict(use_wide=False)),
        ("ABL: - handcrafted features entirely", b, dict(use_hand=False)),
        ("ABL: + node2vec structural embeddings", b_st, dict(use_struct=True)),
        ("ABL: 1 GNN layer instead of 2", b, dict(layers=1)),
        ("ABL: 3 GNN layers instead of 2", b, dict(layers=3)),
    ]:
        if name in out:
            continue
        out[name] = run_gnn(bb, tr, va, te, seed, args, **kw)
        print(f"  {name:<46} AUC={out[name]['auc_roc']:.4f} "
              f"AP={out[name]['auc_pr']:.4f}", flush=True)
        save_partial("ablations", seed, out, args.regime)
    return out


def stage_khop(seed, args):
    out = load_partial("khop", seed, args.regime)
    for k in args.khop_sweep:
        if f"KHOP: ego network radius k={k}" in out:
            continue
        sg, _, inst, _, tr, va, te = setup(seed, args, k_hops=k)
        b = make_batcher(inst, sg, seed, args)
        b.fit_hand_scaler(tr)
        name = f"KHOP: ego network radius k={k}"
        out[name] = run_gnn(b, tr, va, te, seed, args)
        print(f"  {name:<46} AUC={out[name]['auc_roc']:.4f} "
              f"AP={out[name]['auc_pr']:.4f}", flush=True)
        save_partial("khop", seed, out, args.regime)
    return out


# --------------------------------------------------------------------------------------

def stage_report(args):
    runs: dict[str, list] = {}
    meta = {}
    runs_struct: dict[str, list] = {}
    meta_struct = {}
    for path in sorted(glob.glob(os.path.join(PARTIAL, "*.json"))):
        structural = "_structural_" in os.path.basename(path)
        # Regime is encoded in the filename, so the two regimes never get averaged
        # together into one meaningless number.
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            if k == "_meta":
                if structural:
                    meta_struct = v
                else:
                    meta = v
            elif structural:
                runs_struct.setdefault(k, []).append(v)
            else:
                runs.setdefault(k, []).append(v)
    agg = {k: aggregate(v) for k, v in runs.items()}
    n_seeds = max((len(v) for v in runs.values()), default=0)

    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "aggregate": agg, "per_seed": runs,
                   "meta_structural": meta_struct,
                   "aggregate_structural": {k: aggregate(v)
                                            for k, v in runs_struct.items()},
                   "per_seed_structural": runs_struct}, f, indent=2)

    header = ("| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall |\n"
              "|---|---|---|---|---|---|")
    main_keys = [k for k in ORDER if k in agg]
    abl_keys = [k for k in agg if k.startswith("ABL:")]
    khop_keys = sorted(k for k in agg if k.startswith("KHOP:"))

    # "up to" because ablations and the k-hop sweep only ran on seed 0, while the main
    # table ran on three.
    lines = ["# Results", "",
             f"Mean ± sd over up to {n_seeds} seeds. Each seed regenerates the graph, "
             "the cascade and the split, so the spread covers data variance as well as "
             "initialisation variance.", "",
             f"Dataset per seed: `{meta}`", ""]
    for title, keys in [("Regime A (default) - baselines vs proposed model", main_keys),
                        ("Ablations (applied to the full GAT model, regime A)", abl_keys),
                        ("Sensitivity to ego network radius (regime A)", khop_keys)]:
        if keys:
            lines += [f"## {title}", "", header] + [fmt_row(k, agg[k]) for k in keys] + [""]

    if runs_struct:
        agg_s = {k: aggregate(v) for k, v in runs_struct.items()}
        ks = [k for k in ORDER if k in agg_s] + [k for k in agg_s if k not in ORDER]
        lines += ["## Regime B (structure-dominant) - baselines vs proposed model", "",
                  "Same pipeline, same code, a cascade where the structural term "
                  "carries most of the signal and the raw active-neighbour count "
                  "carries almost none.", "",
                  f"Dataset per seed: `{meta_struct}`", "", header]
        lines += [fmt_row(k, agg_s[k]) for k in ks] + [""]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "results.md"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "baselines", "models", "ablations", "khop", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-nodes", type=int, default=8000)
    ap.add_argument("--ego-size", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--regime", default="default", choices=["default", "structural"])
    ap.add_argument("--khop-sweep", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if args.stage == "report":
        stage_report(args); return
    print(f"regime={args.regime}", flush=True)
    # Printed every run. Forgetting which regime a partial result came from is the
    # easiest way to end up comparing numbers that are not comparable.

    fns = {"baselines": stage_baselines, "models": stage_models,
           "ablations": stage_ablations, "khop": stage_khop}
    if args.stage == "all":
        for seed in args.seeds:
            print(f"\n=== seed {seed} ===", flush=True)
            for name in ["baselines", "models", "ablations", "khop"]:
                print(f"-- {name}", flush=True)
                fns[name](seed, args)
        stage_report(args)
    else:
        t = time.time()
        print(f"=== stage={args.stage} seed={args.seed} ===", flush=True)
        fns[args.stage](args.seed, args)
        print(f"done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()

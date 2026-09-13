# Social Influence Prediction with Graph Neural Networks

A full implementation of the DeepInf-style pipeline: sample an ego network around a
user, encode it with a GNN, fuse the embedding with handcrafted features, and predict
whether that user performs the action their neighbours already performed.

The question the project actually answers is not "does the GNN work" but **how much of
the available structural signal does message passing recover, and when is it worth
having at all**. See `REPORT.md` for the finding.

## Quick start

```bash
pip install -r requirements.txt

# full run, ~35 min on one CPU core
python src/experiments.py --stage all --seeds 0 1 2

# or stage by stage (each stage checkpoints to results/partial/ and resumes)
python src/experiments.py --stage baselines --seed 0
python src/experiments.py --stage models    --seed 0
python src/experiments.py --stage ablations --seed 0
python src/experiments.py --stage khop      --seed 0
python src/experiments.py --stage baselines --seed 0 --regime structural
python src/experiments.py --stage models    --seed 0 --regime structural
python src/experiments.py --stage report

python src/analysis.py                      # paired per-seed comparison
python src/export_model.py --regime default # trained checkpoint
```

No GPU needed. No torch-geometric — ego networks are fixed-size, so the whole thing
runs on dense `[B, s, s]` adjacency tensors.

## Layout

```
src/
  data.py          graph generation, cascade simulation, ego sampling, handcrafted features
  models.py        GCN, GAT, instance normalisation, wide-and-deep fusion head
  embeddings.py    DeepWalk and node2vec (skip-gram + negative sampling, from scratch)
  baselines.py     logistic regression, embedding+MLP, diagnostic oracle
  train.py         batching, class-weighted BCE, early stopping
  evaluate.py      AUC-ROC, AUC-PR, F1, precision, recall, threshold selection
  experiments.py   resumable stage runner -> results/
  analysis.py      paired per-seed comparison and headroom accounting
  export_model.py  trains and saves a self-contained checkpoint

results/   results.md, metrics.json, paired_analysis.md
weights/   trained checkpoints (config + scaler + threshold bundled with the state dict)
cache/     generated datasets and embeddings (safe to delete, regenerates)
```

## Pipeline, mapped to the problem statement

| PS section | Where | Note |
|---|---|---|
| (a) ego network sampling | `data.sample_ego_network` | personalised PageRank, k-hop capped, fixed size 50 |
| (b) graph encoding | `models.GCNLayer`, `models.GATLayer` | both implemented, both reported |
| (c) instance normalisation | `models.InstanceNorm` | across nodes within one instance, not across the batch |
| (d) handcrafted features | `data.handcrafted_features` | exactly the block enumerated in the PS |
| (e) prediction head | `models.InfluenceModel.head` | MLP + sigmoid, with a wide/deep skip |
| (f) prediction task | `data.build_instances` | inactive user with ≥1 active neighbour, label = adopts within horizon |
| (g) training | `train.train_model` | BCE with logits, class-weighted, Adam, early stopping |

## Data

The default dataset is synthetic: a Holme–Kim powerlaw-cluster graph with a simulated
generalised-threshold cascade. This is deliberate — a synthetic generator is the only
way to know the ground-truth mechanism, which is what makes the oracle diagnostic in
`REPORT.md` possible at all.

To run on OAG / Digg / Weibo instead, replace `build_graph` and `simulate_cascade` with
a loader that returns (1) an edge list and (2) an action log of `(user, action, time)`.
Everything downstream of `build_instances` is dataset-agnostic.

## Known limitations

Listed honestly in `REPORT.md` §6. The short version: synthetic data, one graph family,
three seeds, no causal identification of influence versus homophily, and a sampler that
may systematically drop the weak ties that carry the most influence.

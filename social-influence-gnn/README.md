# Social Influence Prediction with Graph Neural Networks

Given that some of your friends did something, will you do it too?

That is the whole task. This repo implements the DeepInf pipeline for it: sample an ego network around a user, encode it with a GNN, fuse that embedding with handcrafted features, predict adoption.

The interesting part is not the pipeline. It is the measurement around it. I built the data generator with a known mechanism and a diagnostic oracle, so the question becomes how much of the available structural signal message passing actually recovers.

Short version of the answer: in the default setup the GNN ties logistic regression, in a structure-heavy setup it wins on 3 of 3 seeds, and either way it captures under a fifth of what is there. `REPORT.md` has the numbers and the reason.

## Running it

```bash
pip install -r requirements.txt

# full run, roughly 35 minutes on one CPU core
python src/experiments.py --stage all --seeds 0 1 2
```

Every stage checkpoints to `results/partial/` and resumes, so you can also do it in pieces:

```bash
python src/experiments.py --stage baselines --seed 0
python src/experiments.py --stage models    --seed 0
python src/experiments.py --stage ablations --seed 0
python src/experiments.py --stage khop      --seed 0
python src/experiments.py --stage baselines --seed 0 --regime structural
python src/experiments.py --stage models    --seed 0 --regime structural
python src/experiments.py --stage report

python src/analysis.py                       # paired per-seed comparison
python src/export_model.py --regime default  # trained checkpoint
```

No GPU. No torch-geometric either. Ego networks are fixed-size, so everything runs on dense `[B, s, s]` adjacency tensors and the whole dependency list is five packages.

## Layout

```
src/
  data.py          graph generation, cascade simulation, ego sampling, handcrafted features
  models.py        GCN, GAT, instance normalisation, wide-and-deep fusion head
  embeddings.py    DeepWalk and node2vec, skip-gram with negative sampling, written from scratch
  baselines.py     logistic regression, embedding+MLP, diagnostic oracle
  train.py         batching, class-weighted BCE, early stopping
  evaluate.py      AUC-ROC, AUC-PR, F1, precision, recall, threshold selection
  experiments.py   resumable stage runner
  analysis.py      paired per-seed comparison and headroom accounting
  export_model.py  trains and saves a self-contained checkpoint

results/   results.md, metrics.json, paired_analysis.md
weights/   trained checkpoints, with config, scaler and threshold bundled in
cache/     generated datasets and embeddings, safe to delete, regenerates on next run
```

## Where each part of the problem statement lives

| PS section | Code | Note |
|---|---|---|
| (a) ego network sampling | `data.sample_ego_network` | personalised PageRank, k-hop capped, fixed size 50 |
| (b) graph encoding | `models.GCNLayer`, `models.GATLayer` | both built, both reported |
| (c) instance normalisation | `models.InstanceNorm` | across nodes inside one instance, not across the batch |
| (d) handcrafted features | `data.handcrafted_features` | exactly the block the PS enumerates |
| (e) prediction head | `models.InfluenceModel.head` | MLP and sigmoid, with a wide/deep skip |
| (f) prediction task | `data.build_instances` | inactive user with at least one active neighbour, label is adoption within the horizon |
| (g) training | `train.train_model` | BCE with logits, class-weighted, Adam, early stopping |

## About the data

The default dataset is synthetic: a Holme-Kim powerlaw-cluster graph with a simulated generalised-threshold cascade.

That is a deliberate choice. A synthetic generator is the only way to know the true mechanism, and knowing the true mechanism is what makes the oracle diagnostic in `REPORT.md` possible.

To run on OAG, Digg or Weibo instead, swap `build_graph` and `simulate_cascade` for a loader that returns an edge list and an action log of `(user, action, time)`. Everything downstream of `build_instances` is dataset-agnostic.

## Limitations

Written out properly in `REPORT.md` §6. The short version: synthetic data, one graph family, three seeds, a sampler that may drop exactly the weak ties that matter most, and no causal separation of influence from homophily.

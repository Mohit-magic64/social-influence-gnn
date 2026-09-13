"""
embeddings.py
-------------
Unsupervised structural node embeddings: DeepWalk (uniform walks) and node2vec
(second-order biased walks), both trained with skip-gram + negative sampling.

They serve two purposes in this project:
  1. as a standalone baseline (embeddings -> MLP, no graph reasoning at predict time)
  2. as an extra input channel inside the ego network, which is what DeepInf does -
     the walk embedding carries global position in the graph that a 2-hop ego network
     cannot see on its own.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Walk generation
# --------------------------------------------------------------------------------------

def deepwalk_walks(adj_list, n_walks: int = 6, walk_len: int = 24, seed: int = 0):
    """Uniform random walks. DeepWalk = 'sentences of node ids' fed to word2vec."""
    rng = np.random.default_rng(seed)
    N = len(adj_list)
    walks = np.zeros((n_walks * N, walk_len), dtype=np.int32)
    row = 0
    for _ in range(n_walks):
        for start in rng.permutation(N):
            cur = int(start)
            walks[row, 0] = cur
            for t in range(1, walk_len):
                nbrs = adj_list[cur]
                if nbrs.size == 0:
                    walks[row, t] = cur
                    continue
                cur = int(nbrs[rng.integers(nbrs.size)])
                walks[row, t] = cur
            row += 1
    return walks


def node2vec_walks(adj_list, n_walks: int = 6, walk_len: int = 24, p: float = 1.0,
                   q: float = 0.5, seed: int = 0):
    """
    Second-order biased walk.
      p (return parameter): high p discourages immediately walking back.
      q (in-out parameter): q < 1 biases outward -> DFS-like -> captures communities
                            (homophily). q > 1 biases inward -> BFS-like -> captures
                            structural roles.
    We use q = 0.5 because influence spread is a homophily phenomenon: who you are
    close to matters more than what role you play.
    """
    rng = np.random.default_rng(seed + 13)
    N = len(adj_list)
    nbr_sets = [set(a.tolist()) for a in adj_list]
    walks = np.zeros((n_walks * N, walk_len), dtype=np.int32)
    row = 0
    for _ in range(n_walks):
        for start in rng.permutation(N):
            cur, prev = int(start), -1
            walks[row, 0] = cur
            for t in range(1, walk_len):
                nbrs = adj_list[cur]
                if nbrs.size == 0:
                    walks[row, t] = cur
                    continue
                if prev < 0:
                    nxt = int(nbrs[rng.integers(nbrs.size)])
                else:
                    w = np.empty(nbrs.size, dtype=np.float64)
                    for i, x in enumerate(nbrs):
                        xi = int(x)
                        if xi == prev:
                            w[i] = 1.0 / p
                        elif xi in nbr_sets[prev]:
                            w[i] = 1.0
                        else:
                            w[i] = 1.0 / q
                    w /= w.sum()
                    nxt = int(nbrs[rng.choice(nbrs.size, p=w)])
                prev, cur = cur, nxt
                walks[row, t] = cur
            row += 1
    return walks


# --------------------------------------------------------------------------------------
# Skip-gram with negative sampling
# --------------------------------------------------------------------------------------

def _pairs_from_walks(walks: np.ndarray, window: int = 3, rng=None,
                      max_pairs: int = 5_000_000):
    """
    Kept as int32 and hard-capped. The naive version materialises
    2 * window * n_walks * n_nodes * walk_len pairs as int64, which is several GB on a
    modest graph — it is the first thing that falls over when you scale this up.
    """
    rng = rng or np.random.default_rng(0)
    L = walks.shape[1]
    centers, contexts = [], []
    for off in range(1, window + 1):
        centers.append(walks[:, :L - off].ravel())
        contexts.append(walks[:, off:].ravel())
        centers.append(walks[:, off:].ravel())
        contexts.append(walks[:, :L - off].ravel())
    c = np.concatenate(centers).astype(np.int32)
    x = np.concatenate(contexts).astype(np.int32)
    if c.size > max_pairs:
        sel = rng.choice(c.size, size=max_pairs, replace=False)
        c, x = c[sel], x[sel]
    return c, x


def train_sgns(walks: np.ndarray, n_nodes: int, dim: int = 64, window: int = 3,
               n_neg: int = 5, epochs: int = 2, batch: int = 4096, lr: float = 0.01,
               seed: int = 0, device: str = "cpu", verbose: bool = False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centers, contexts = _pairs_from_walks(walks, window, rng)

    # Negative sampling distribution: unigram^0.75, the word2vec convention. It pulls
    # probability mass off hubs so that low-degree nodes still act as informative
    # negatives.
    counts = np.bincount(walks.ravel(), minlength=n_nodes).astype(np.float64)
    noise = counts ** 0.75
    noise = torch.tensor(noise / noise.sum(), dtype=torch.float, device=device)

    emb_in = nn.Embedding(n_nodes, dim).to(device)
    emb_out = nn.Embedding(n_nodes, dim).to(device)
    nn.init.uniform_(emb_in.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(emb_out.weight)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)

    n_pairs = centers.size

    for ep in range(epochs):
        perm = rng.permutation(n_pairs)
        total, nb = 0.0, 0
        for i in range(0, n_pairs, batch):
            idx = perm[i:i + batch]
            c = torch.from_numpy(centers[idx].astype(np.int64)).to(device)
            x = torch.from_numpy(contexts[idx].astype(np.int64)).to(device)
            neg = torch.multinomial(noise, c.numel() * n_neg, replacement=True)
            neg = neg.view(c.numel(), n_neg)

            v = emb_in(c)                                   # [B, d]
            pos_score = (v * emb_out(x)).sum(-1)
            neg_score = torch.bmm(emb_out(neg), v.unsqueeze(-1)).squeeze(-1)

            loss = (-torch.nn.functional.logsigmoid(pos_score).mean()
                    - torch.nn.functional.logsigmoid(-neg_score).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if verbose:
            print(f"    sgns epoch {ep + 1}/{epochs}  loss={total / max(nb, 1):.4f}")

    return emb_in.weight.detach().cpu().numpy()


def get_embeddings(adj_list, n_nodes: int, kind: str = "deepwalk", dim: int = 64,
                   cache_dir: str = "cache", seed: int = 0, rebuild: bool = False,
                   verbose: bool = False) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{kind}_d{dim}_n{n_nodes}_seed{seed}.npy")
    if os.path.exists(path) and not rebuild:
        return np.load(path)
    if verbose:
        print(f"  generating {kind} walks ...")
    walks = (deepwalk_walks(adj_list, seed=seed) if kind == "deepwalk"
             else node2vec_walks(adj_list, seed=seed))
    emb = train_sgns(walks, n_nodes, dim=dim, seed=seed, verbose=verbose)
    # L2-normalise: keeps the scale of this input channel comparable to the other
    # node features so the input projection is not dominated by one block.
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    np.save(path, emb.astype(np.float32))
    return emb.astype(np.float32)

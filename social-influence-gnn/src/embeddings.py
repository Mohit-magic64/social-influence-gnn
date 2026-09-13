"""
DeepWalk and node2vec: unsupervised node embeddings from random walks.

Used two ways here: as a standalone baseline, and as an optional extra input channel
inside the ego network (which the ablation shows actually hurts).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


# ======================================================================================
# Walk generation
# ======================================================================================

def deepwalk_walks(adj_list, n_walks: int = 6, walk_len: int = 24, seed: int = 0):
    """Uniform random walks. DeepWalk treats each walk as a sentence for word2vec."""
    rng = np.random.default_rng(seed)
    N = len(adj_list)
    walks = np.zeros((n_walks * N, walk_len), dtype=np.int32)
    row = 0
    for _ in range(n_walks):
        for start in rng.permutation(N):
            # Every node starts a walk, so even low-degree nodes get an embedding.
            cur = int(start)
            walks[row, 0] = cur
            for t in range(1, walk_len):
                nbrs = adj_list[cur]
                if nbrs.size == 0:
                    walks[row, t] = cur     # isolated node just stays put
                    continue
                cur = int(nbrs[rng.integers(nbrs.size)])
                walks[row, t] = cur
            row += 1
    return walks


def node2vec_walks(adj_list, n_walks: int = 6, walk_len: int = 24, p: float = 1.0,
                   q: float = 0.5, seed: int = 0):
    """Second-order biased walks."""
    # p is the return parameter: high p discourages stepping straight back.
    # q is the in-out parameter: q < 1 biases outward (DFS-like, captures communities),
    # q > 1 biases inward (BFS-like, captures structural roles).
    # q = 0.5 here because influence spread is a homophily phenomenon. Who you are close
    # to matters more than what structural role you play.
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
                    # First step has no previous node, so no bias to apply.
                else:
                    w = np.empty(nbrs.size, dtype=np.float64)
                    for i, x in enumerate(nbrs):
                        xi = int(x)
                        if xi == prev:
                            w[i] = 1.0 / p       # stepping back
                        elif xi in nbr_sets[prev]:
                            w[i] = 1.0           # stays within one hop of prev
                        else:
                            w[i] = 1.0 / q       # moves further out
                    w /= w.sum()
                    nxt = int(nbrs[rng.choice(nbrs.size, p=w)])
                prev, cur = cur, nxt
                walks[row, t] = cur
            row += 1
    return walks


# ======================================================================================
# Skip-gram with negative sampling
# ======================================================================================

def _pairs_from_walks(walks: np.ndarray, window: int = 3, rng=None,
                      max_pairs: int = 5_000_000):
    """(center, context) pairs from every walk, int32 and hard-capped."""
    # The naive version builds 2 * window * n_walks * N * walk_len pairs as int64, which
    # is several GB on a modest graph. It is the first thing that falls over when you
    # scale this up, and it did fall over here.
    rng = rng or np.random.default_rng(0)
    L = walks.shape[1]
    centers, contexts = [], []
    for off in range(1, window + 1):
        centers.append(walks[:, :L - off].ravel())
        contexts.append(walks[:, off:].ravel())
        centers.append(walks[:, off:].ravel())
        contexts.append(walks[:, :L - off].ravel())
        # Both directions, since the skip-gram context window is symmetric.
    c = np.concatenate(centers).astype(np.int32)
    x = np.concatenate(contexts).astype(np.int32)
    if c.size > max_pairs:
        sel = rng.choice(c.size, size=max_pairs, replace=False)
        c, x = c[sel], x[sel]
    return c, x


def train_sgns(walks: np.ndarray, n_nodes: int, dim: int = 64, window: int = 3,
               n_neg: int = 5, epochs: int = 2, batch: int = 4096, lr: float = 0.01,
               seed: int = 0, device: str = "cpu", verbose: bool = False):
    """Skip-gram with negative sampling. Returns the input embedding matrix."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centers, contexts = _pairs_from_walks(walks, window, rng)

    counts = np.bincount(walks.ravel(), minlength=n_nodes).astype(np.float64)
    noise = counts ** 0.75
    noise = torch.tensor(noise / noise.sum(), dtype=torch.float, device=device)
    # The 0.75 exponent is the word2vec convention. It flattens the distribution so hubs
    # are not always the negative, and low-degree nodes still act as useful negatives.

    emb_in = nn.Embedding(n_nodes, dim).to(device)
    emb_out = nn.Embedding(n_nodes, dim).to(device)
    nn.init.uniform_(emb_in.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(emb_out.weight)
    # Two separate matrices, one for the node as center and one as context. Sharing them
    # makes every node similar to itself and degrades the embeddings.
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)

    n_pairs = centers.size

    for ep in range(epochs):
        perm = rng.permutation(n_pairs)
        total, nb = 0.0, 0
        for i in range(0, n_pairs, batch):
            idx = perm[i:i + batch]
            c = torch.from_numpy(centers[idx].astype(np.int64)).to(device)
            x = torch.from_numpy(contexts[idx].astype(np.int64)).to(device)
            # Converted per batch. Holding all pairs as int64 tensors is what blew up
            # memory in the first version.
            neg = torch.multinomial(noise, c.numel() * n_neg, replacement=True)
            neg = neg.view(c.numel(), n_neg)

            v = emb_in(c)                                   # [B, d]
            pos_score = (v * emb_out(x)).sum(-1)
            neg_score = torch.bmm(emb_out(neg), v.unsqueeze(-1)).squeeze(-1)
            # bmm gives each row its own negatives in one batched matmul.

            loss = (-torch.nn.functional.logsigmoid(pos_score).mean()
                    - torch.nn.functional.logsigmoid(-neg_score).mean())
            # Push observed pairs together, push sampled pairs apart. logsigmoid rather
            # than log(sigmoid(.)) because it is numerically stable at large magnitudes.
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if verbose:
            print(f"    sgns epoch {ep + 1}/{epochs}  loss={total / max(nb, 1):.4f}")

    return emb_in.weight.detach().cpu().numpy()


def get_embeddings(adj_list, n_nodes: int, kind: str = "deepwalk", dim: int = 64,
                   cache_dir: str = "cache", seed: int = 0, rebuild: bool = False,
                   verbose: bool = False) -> np.ndarray:
    """Cached embeddings. kind is 'deepwalk' or 'node2vec'."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{kind}_d{dim}_n{n_nodes}_seed{seed}.npy")
    if os.path.exists(path) and not rebuild:
        return np.load(path)
        # Cached per (kind, dim, n_nodes, seed). The graph depends only on the seed, so
        # both cascade regimes reuse the same embeddings.
    if verbose:
        print(f"  generating {kind} walks ...")
    walks = (deepwalk_walks(adj_list, seed=seed) if kind == "deepwalk"
             else node2vec_walks(adj_list, seed=seed))
    emb = train_sgns(walks, n_nodes, dim=dim, seed=seed, verbose=verbose)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    # L2-normalise so this 64-dim block does not dominate the 7 raw node features when
    # they are concatenated and fed to the input projection.
    np.save(path, emb.astype(np.float32))
    return emb.astype(np.float32)

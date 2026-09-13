"""
data.py
-------
Builds the supervised dataset for social influence prediction.

Three stages:
  1. build_graph()        -> a social graph with realistic degree/clustering structure
                             and per-node latent attributes
  2. simulate_cascade()   -> an observed diffusion process over that graph
  3. build_instances()    -> DeepInf-style labelled instances:
                             (ego network, neighbour action states) -> did the ego act?

The key modelling decision is in simulate_cascade(). The probability that a user
adopts depends on THREE things:
    - how many of their neighbours already adopted   (a count)
    - what fraction of their neighbours adopted      (a ratio)
    - how many *disconnected groups* those adopters form  (structural diversity)

The third term is the interesting one. It is a real empirical finding (Ugander et al.,
PNAS 2012): three adopters who all know each other are far less persuasive than three
adopters from three separate parts of your life. It is also, deliberately, a quantity
that CANNOT be read off the handcrafted feature list in the problem statement
(degree, #active neighbours, fraction active). A model has to look at the wiring
*between* your neighbours to recover it — which is exactly what a GNN can do and a
logistic regression on ego-level counts cannot.

That is what makes the comparison in this project meaningful rather than decorative.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------------------
# 1. Graph + node attributes
# --------------------------------------------------------------------------------------

@dataclass
class SocialGraph:
    G: nx.Graph
    attrs: np.ndarray          # [N, A] observable user attributes
    susceptibility: np.ndarray  # [N]   latent trait, NOT observable by the model
    adj_list: list = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.G.number_of_nodes()


def build_graph(n_nodes: int = 6000, m: int = 5, p_triangle: float = 0.40,
                seed: int = 0) -> SocialGraph:
    """
    Holme-Kim powerlaw-cluster graph: preferential attachment (heavy-tailed degree,
    like a real social network) plus an explicit triangle-closure step (high clustering,
    also like a real social network). Erdos-Renyi would give neither.

    Observable attributes per user (these become the 'user attributes' in the
    handcrafted feature block and the node features inside the ego network):
        0: log activity level     - how much the user posts
        1: tenure                 - how long they have been on the platform
        2: historical action rate  - how often they have adopted things before
    """
    rng = np.random.default_rng(seed)
    G = nx.powerlaw_cluster_graph(n_nodes, m, p_triangle, seed=seed)

    activity = rng.lognormal(mean=0.0, sigma=0.8, size=n_nodes)
    tenure = rng.uniform(0.0, 1.0, size=n_nodes)
    hist_rate = rng.beta(2.0, 6.0, size=n_nodes)

    attrs = np.stack([np.log1p(activity), tenure, hist_rate], axis=1).astype(np.float32)

    # Latent susceptibility: partly explained by observables, partly not.
    # The unexplained part is irreducible noise and caps how good any model can get.
    lin = 0.9 * (hist_rate - hist_rate.mean()) / hist_rate.std() \
        + 0.4 * (np.log1p(activity) - np.log1p(activity).mean()) / np.log1p(activity).std()
    susceptibility = lin + rng.normal(0, 1.0, size=n_nodes)

    adj_list = [np.array(sorted(G.neighbors(i)), dtype=np.int32) for i in range(n_nodes)]
    return SocialGraph(G=G, attrs=attrs, susceptibility=susceptibility, adj_list=adj_list)


# --------------------------------------------------------------------------------------
# 2. Cascade simulation
# --------------------------------------------------------------------------------------

CASCADE_COEFS = dict(
    intercept=-3.20,
    n_active=0.30,        # log(1 + #active neighbours)
    frac_active=1.25,     # #active / degree
    diversity=1.20,       # #connected components among active neighbours  <-- structural
    susceptibility=0.55,  # latent user trait
    log_degree=-0.30,     # popular users are harder to move per-neighbour
)


STRUCTURE_DOMINANT = dict(CASCADE_COEFS)
STRUCTURE_DOMINANT.update(intercept=-4.20, n_active=0.05, frac_active=0.35,
                          diversity=2.60, susceptibility=0.40)

REGIMES = {"default": CASCADE_COEFS, "structural": STRUCTURE_DOMINANT}


def _active_component_count(active_nbrs: np.ndarray, adj_sets: list) -> int:
    """Number of connected components in the subgraph induced on the active neighbours."""
    k = len(active_nbrs)
    if k <= 1:
        return k
    idx = {u: i for i, u in enumerate(active_nbrs)}
    seen = np.zeros(k, dtype=bool)
    comps = 0
    for start in range(k):
        if seen[start]:
            continue
        comps += 1
        stack = [start]
        seen[start] = True
        while stack:
            cur = stack.pop()
            for nb in adj_sets[active_nbrs[cur]]:
                j = idx.get(nb)
                if j is not None and not seen[j]:
                    seen[j] = True
                    stack.append(j)
    return comps


def simulate_cascade(sg: SocialGraph, n_rounds: int = 10, seed_frac: float = 0.012,
                     seed: int = 0, coefs: dict | None = None):
    """
    A generalised-threshold diffusion process. At each round every inactive user draws
    an adoption probability from a logistic function of their local state, and flips a
    coin. Returns the round index at which each node activated (-1 = never).
    """
    coefs = coefs or CASCADE_COEFS
    rng = np.random.default_rng(seed + 991)
    N = sg.n_nodes
    adj_sets = [set(a.tolist()) for a in sg.adj_list]
    degree = np.array([len(a) for a in sg.adj_list], dtype=np.float32)
    log_deg = np.log1p(degree)

    active_at = np.full(N, -1, dtype=np.int32)

    # Seed users: degree-biased, as real cascades usually start with hubs.
    seed_p = degree / degree.sum()
    n_seed = max(8, int(seed_frac * N))
    seeds = rng.choice(N, size=n_seed, replace=False, p=seed_p)
    active_at[seeds] = 0

    for t in range(1, n_rounds + 1):
        active = active_at >= 0
        candidates = np.where(~active)[0]
        newly = []
        for v in candidates:
            nbrs = sg.adj_list[v]
            if nbrs.size == 0:
                continue
            act_nbrs = nbrs[active[nbrs]]
            n_a = act_nbrs.size
            if n_a == 0:
                continue
            div = _active_component_count(act_nbrs, adj_sets)
            h = (coefs["intercept"]
                 + coefs["n_active"] * np.log1p(n_a)
                 + coefs["frac_active"] * (n_a / max(nbrs.size, 1))
                 + coefs["diversity"] * np.log1p(div)
                 + coefs["susceptibility"] * sg.susceptibility[v]
                 + coefs["log_degree"] * log_deg[v])
            if rng.random() < 1.0 / (1.0 + np.exp(-h)):
                newly.append(v)
        if not newly:
            break
        active_at[np.array(newly, dtype=np.int32)] = t

    return active_at


# --------------------------------------------------------------------------------------
# 3. Ego network sampling
# --------------------------------------------------------------------------------------

def _khop_pool(v: int, adj_list: list, k: int, cap: int, rng) -> np.ndarray:
    """BFS out to k hops, capped. Returns candidate node ids including v at position 0."""
    seen = {v}
    order = [v]
    frontier = [v]
    for _ in range(k):
        nxt = []
        for u in frontier:
            nbrs = adj_list[u]
            if nbrs.size and len(seen) + nbrs.size > cap * 3:
                nbrs = rng.choice(nbrs, size=min(nbrs.size, 60), replace=False)
            for w in nbrs:
                w = int(w)
                if w not in seen:
                    seen.add(w)
                    order.append(w)
                    nxt.append(w)
            if len(order) >= cap * 3:
                break
        frontier = nxt
        if not frontier or len(order) >= cap * 3:
            break
    return np.array(order[: cap * 3], dtype=np.int32)


def _personalised_pagerank(pool: np.ndarray, adj_list: list, alpha: float = 0.15,
                           n_iter: int = 25) -> np.ndarray:
    """
    Restart probability alpha, 25 power iterations, restricted to the candidate pool.

    Why PPR instead of literally running random walks: the stationary distribution of a
    random-walk-with-restart IS the expected visit frequency of that walk. Computing it
    directly is deterministic and ~100x faster than sampling walks, with no variance.
    """
    n = pool.size
    idx = {int(u): i for i, u in enumerate(pool)}
    P = np.zeros((n, n), dtype=np.float32)
    for i, u in enumerate(pool):
        nbrs = adj_list[u]
        local = [idx[int(w)] for w in nbrs if int(w) in idx]
        if local:
            P[i, local] = 1.0 / len(local)
    e = np.zeros(n, dtype=np.float32)
    e[0] = 1.0
    r = e.copy()
    for _ in range(n_iter):
        r = (1.0 - alpha) * (P.T @ r) + alpha * e
    return r


def sample_ego_network(v: int, adj_list: list, size: int, k_hops: int,
                       rng, strategy: str = "ppr"):
    """
    Returns (node_ids [<=size], local adjacency [s, s] uint8) with the ego at index 0.

    Fixed size is not cosmetic: it lets every instance be stored and batched as a dense
    [B, s, s] tensor, which is what makes the GNN trainable at scale without a
    sparse-graph library.
    """
    pool = _khop_pool(v, adj_list, k_hops, cap=size, rng=rng)
    if pool.size > size:
        if strategy == "ppr":
            scores = _personalised_pagerank(pool, adj_list)
            scores[0] = np.inf                      # ego always kept
            keep = np.argsort(-scores)[:size]
            keep = np.sort(keep)
            keep = np.concatenate([[0], keep[keep != 0]])[:size]
        else:                                        # plain BFS order
            keep = np.arange(size)
        nodes = pool[keep]
    else:
        nodes = pool

    s = nodes.size
    idx = {int(u): i for i, u in enumerate(nodes)}
    adj = np.zeros((s, s), dtype=np.uint8)
    for i, u in enumerate(nodes):
        for w in adj_list[u]:
            j = idx.get(int(w))
            if j is not None:
                adj[i, j] = 1
                adj[j, i] = 1
    np.fill_diagonal(adj, 0)
    return nodes.astype(np.int32), adj


# --------------------------------------------------------------------------------------
# 4. Handcrafted features
# --------------------------------------------------------------------------------------

# Exactly the block enumerated in section 2.4 of the problem statement: user
# attributes, structural features, local action statistics. Nothing else.
PS_HAND_FEATURES = [
    "log_degree", "clustering_coef", "log_n_active", "frac_active",
    "influence_ratio", "mean_active_nbr_logdeg", "max_active_nbr_logdeg",
    "attr_log_activity", "attr_tenure", "attr_hist_rate",
]

# Computed and stored, but kept OUT of the default feature set. A 2-hop active count is
# not in the problem statement's list, and it is a hand-built proxy for exactly the
# structural signal the GNN is supposed to learn. Handing it to the baseline for free
# would answer a different question. It is reported separately as a strengthened
# baseline so the comparison stays visible rather than quietly favourable.
EXTRA_HAND_FEATURES = ["log_active_2hop"]

HAND_FEATURE_NAMES = PS_HAND_FEATURES + EXTRA_HAND_FEATURES
N_PS_FEATURES = len(PS_HAND_FEATURES)


def handcrafted_features(v: int, sg: SocialGraph, active: np.ndarray,
                         clustering: np.ndarray) -> np.ndarray:
    nbrs = sg.adj_list[v]
    deg = max(nbrs.size, 1)
    act_mask = active[nbrs]
    act_nbrs = nbrs[act_mask]
    n_a = act_nbrs.size

    nbr_deg = np.array([sg.adj_list[u].size for u in nbrs], dtype=np.float32)
    act_deg = nbr_deg[act_mask]
    influence_ratio = (act_deg.sum() / nbr_deg.sum()) if nbr_deg.sum() > 0 else 0.0

    two_hop_active = 0
    for u in nbrs:
        two_hop_active += int(active[sg.adj_list[u]].sum())

    return np.array([
        np.log1p(deg),
        clustering[v],
        np.log1p(n_a),
        n_a / deg,
        influence_ratio,
        np.log1p(act_deg).mean() if n_a else 0.0,
        np.log1p(act_deg).max() if n_a else 0.0,
        sg.attrs[v, 0], sg.attrs[v, 1], sg.attrs[v, 2],
        np.log1p(two_hop_active),
    ], dtype=np.float32)


# --------------------------------------------------------------------------------------
# 5. Instance construction
# --------------------------------------------------------------------------------------

@dataclass
class InstanceSet:
    node_ids: np.ndarray     # [M, s] int32   (ego at column 0)
    adj: np.ndarray          # [M, s, s] uint8
    active: np.ndarray       # [M, s] uint8   neighbour action states at observation time
    valid: np.ndarray        # [M, s] uint8   padding mask
    hand: np.ndarray         # [M, H] float32
    y: np.ndarray            # [M] int8
    ego: np.ndarray          # [M] int32
    round_idx: np.ndarray    # [M] int16  observation round this instance was cut at
    hand_names: list


def build_instances(sg: SocialGraph, active_at: np.ndarray, observe_round: int,
                    horizon: int, ego_size: int = 50, k_hops: int = 2,
                    strategy: str = "ppr", max_instances: int | None = None,
                    seed: int = 0) -> InstanceSet:
    """
    Freeze the world at `observe_round`. Every user who is still inactive but has at
    least one active neighbour becomes a candidate instance. The label is whether they
    adopt within the next `horizon` rounds.

    The ego's own action state is 0 by construction here, but we mask it explicitly
    anyway in the model. Leaking the ego's future state into the input is the single
    most common way this task gets silently broken.
    """
    rng = np.random.default_rng(seed + 77)
    active = (active_at >= 0) & (active_at <= observe_round)
    clustering = np.zeros(sg.n_nodes, dtype=np.float32)
    for node, c in nx.clustering(sg.G).items():
        clustering[node] = c

    candidates = []
    for v in range(sg.n_nodes):
        if active[v]:
            continue
        nbrs = sg.adj_list[v]
        if nbrs.size == 0 or not active[nbrs].any():
            continue
        candidates.append(v)
    candidates = np.array(candidates, dtype=np.int32)
    rng.shuffle(candidates)
    if max_instances is not None:
        candidates = candidates[:max_instances]

    M, s = candidates.size, ego_size
    node_ids = np.zeros((M, s), dtype=np.int32)
    adjs = np.zeros((M, s, s), dtype=np.uint8)
    acts = np.zeros((M, s), dtype=np.uint8)
    valid = np.zeros((M, s), dtype=np.uint8)
    hand = np.zeros((M, len(HAND_FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(M, dtype=np.int8)

    for i, v in enumerate(candidates):
        nodes, a = sample_ego_network(v, sg.adj_list, s, k_hops, rng, strategy)
        n = nodes.size
        node_ids[i, :n] = nodes
        adjs[i, :n, :n] = a
        acts[i, :n] = active[nodes].astype(np.uint8)
        acts[i, 0] = 0                     # explicit ego mask
        valid[i, :n] = 1
        hand[i] = handcrafted_features(v, sg, active, clustering)
        y[i] = 1 if (0 <= active_at[v] <= observe_round + horizon) else 0

    rounds = np.full(M, observe_round, dtype=np.int16)
    return InstanceSet(node_ids, adjs, acts, valid, hand, y, candidates, rounds,
                       list(HAND_FEATURE_NAMES))


# --------------------------------------------------------------------------------------
# 6. End-to-end builder with caching
# --------------------------------------------------------------------------------------

def concat_instances(parts: list[InstanceSet]) -> InstanceSet:
    return InstanceSet(
        np.concatenate([p.node_ids for p in parts]),
        np.concatenate([p.adj for p in parts]),
        np.concatenate([p.active for p in parts]),
        np.concatenate([p.valid for p in parts]),
        np.concatenate([p.hand for p in parts]),
        np.concatenate([p.y for p in parts]),
        np.concatenate([p.ego for p in parts]),
        np.concatenate([p.round_idx for p in parts]),
        parts[0].hand_names,
    )


def build_dataset(cache_dir: str = "cache", seed: int = 0, n_nodes: int = 20000,
                  ego_size: int = 50, k_hops: int = 2, strategy: str = "ppr",
                  observe_rounds=(4, 5, 6), horizon: int = 2,
                  max_per_round: int | None = 4000, rebuild: bool = False,
                  regime: str = "default"):
    """
    Instances are drawn at several observation times, as DeepInf does. One snapshot
    gives too few labelled users and biases the dataset towards one phase of the
    cascade (early adopters look different from late ones).
    """
    os.makedirs(cache_dir, exist_ok=True)
    rounds_tag = "-".join(map(str, observe_rounds))
    tag = (f"n{n_nodes}_s{ego_size}_k{k_hops}_{strategy}_o{rounds_tag}"
           f"_h{horizon}_{regime}_seed{seed}")
    path = os.path.join(cache_dir, f"ds_{tag}.pkl")
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)

    sg = build_graph(n_nodes=n_nodes, seed=seed)
    active_at = simulate_cascade(sg, seed=seed, coefs=REGIMES[regime])
    parts, snaps = [], {}
    for r in observe_rounds:
        parts.append(build_instances(sg, active_at, r, horizon, ego_size, k_hops,
                                     strategy, max_per_round, seed + r))
        snaps[r] = (active_at >= 0) & (active_at <= r)
    inst = concat_instances(parts)
    payload = (sg, active_at, inst, snaps)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


def split_indices(inst: InstanceSet, seed: int = 0, fracs=(0.6, 0.15, 0.25)):
    """
    Split by EGO USER, not by row.

    A user can appear as an instance at more than one observation round. If you split
    rows at random, the same person shows up in train and test with an almost identical
    ego network, and the test score is inflated. Grouping by ego closes that door.
    Peripheral nodes can still be shared between splits — that is inherent to any graph
    task and true of DeepInf as well.
    """
    rng = np.random.default_rng(seed + 4242)
    users = np.unique(inst.ego)
    perm = rng.permutation(users)
    a = int(fracs[0] * users.size)
    b = a + int(fracs[1] * users.size)
    groups = {"train": set(perm[:a].tolist()), "val": set(perm[a:b].tolist()),
              "test": set(perm[b:].tolist())}
    out = []
    for key in ("train", "val", "test"):
        g = groups[key]
        out.append(np.array([i for i, e in enumerate(inst.ego) if int(e) in g],
                            dtype=np.int64))
    return tuple(out)

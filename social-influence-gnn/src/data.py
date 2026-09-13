"""
Builds the labelled dataset: graph -> cascade -> ego-network instances.

The label for user v is "did v adopt in the next few rounds", given only the
action states of the people around them at observation time.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


# ======================================================================================
# 1. Graph + node attributes
# ======================================================================================

@dataclass
class SocialGraph:
    G: nx.Graph
    attrs: np.ndarray           # [N, 3] attributes the model is allowed to see
    susceptibility: np.ndarray  # [N] latent trait, NOT given to the model
    adj_list: list = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.G.number_of_nodes()


def build_graph(n_nodes: int = 6000, m: int = 5, p_triangle: float = 0.40,
                seed: int = 0) -> SocialGraph:
    """Holme-Kim powerlaw-cluster graph plus per-user attributes."""
    rng = np.random.default_rng(seed)

    G = nx.powerlaw_cluster_graph(n_nodes, m, p_triangle, seed=seed)
    # Two properties we need and Erdos-Renyi would not give us: a heavy-tailed degree
    # distribution (from preferential attachment) and high clustering (from the
    # triangle-closure step, p_triangle). Clustering matters a lot here, because the
    # whole structural signal lives in whether your neighbours know each other.

    activity = rng.lognormal(mean=0.0, sigma=0.8, size=n_nodes)
    # Lognormal, not normal: a few users post far more than everyone else.
    tenure = rng.uniform(0.0, 1.0, size=n_nodes)
    hist_rate = rng.beta(2.0, 6.0, size=n_nodes)
    # Beta(2,6) is skewed low. Most people rarely adopt things.

    attrs = np.stack([np.log1p(activity), tenure, hist_rate], axis=1).astype(np.float32)

    lin = 0.9 * (hist_rate - hist_rate.mean()) / hist_rate.std() \
        + 0.4 * (np.log1p(activity) - np.log1p(activity).mean()) / np.log1p(activity).std()
    susceptibility = lin + rng.normal(0, 1.0, size=n_nodes)
    # Susceptibility is partly explained by the visible attributes and partly not.
    # That unexplained N(0,1) term is irreducible noise: it is why no model, not even
    # the oracle, gets anywhere near AUC 1.0 on this data.

    adj_list = [np.array(sorted(G.neighbors(i)), dtype=np.int32) for i in range(n_nodes)]
    # Numpy adjacency lists rather than calling nx.neighbors() in hot loops. The cascade
    # touches every node every round, and networkx lookups there are painfully slow.
    return SocialGraph(G=G, attrs=attrs, susceptibility=susceptibility, adj_list=adj_list)


# ======================================================================================
# 2. Cascade simulation
# ======================================================================================

# Adoption is a logistic function of the local state. Three of these terms describe the
# neighbourhood, and the split between them is the point of the whole project:
#   n_active    - how many neighbours adopted           (a plain count)
#   frac_active - what share of neighbours adopted      (a ratio)
#   diversity   - how many disconnected groups they form (structural)
#
# Only the third needs the wiring BETWEEN neighbours, so only the third is out of reach
# of the handcrafted feature list in the problem statement. It is also a real effect:
# Ugander et al. (PNAS 2012) found three adopters from three separate social circles are
# much more persuasive than three adopters who all know each other.
CASCADE_COEFS = dict(
    intercept=-3.20,
    n_active=0.30,
    frac_active=1.25,
    diversity=1.20,
    susceptibility=0.55,
    log_degree=-0.30,     # high-degree users are harder to move per neighbour
)

# Same process, structural term turned up and the raw count turned almost off.
# Running both regimes is what stops this from being a rigged demo: report one regime
# and you can always pick the one that flatters your model.
STRUCTURE_DOMINANT = dict(CASCADE_COEFS)
STRUCTURE_DOMINANT.update(intercept=-4.20, n_active=0.05, frac_active=0.35,
                          diversity=2.60, susceptibility=0.40)

REGIMES = {"default": CASCADE_COEFS, "structural": STRUCTURE_DOMINANT}


def _active_component_count(active_nbrs: np.ndarray, adj_sets: list) -> int:
    """How many connected components the active neighbours form among themselves."""
    k = len(active_nbrs)
    if k <= 1:
        return k
    idx = {u: i for i, u in enumerate(active_nbrs)}
    # Map global node ids to 0..k-1 so the DFS below stays inside the induced subgraph.
    seen = np.zeros(k, dtype=bool)
    comps = 0
    for start in range(k):
        if seen[start]:
            continue
        comps += 1          # every unvisited start is a new component
        stack = [start]
        seen[start] = True
        while stack:
            cur = stack.pop()
            for nb in adj_sets[active_nbrs[cur]]:
                j = idx.get(nb)
                if j is not None and not seen[j]:
                    # j is not None filters out neighbours outside the active set.
                    seen[j] = True
                    stack.append(j)
    return comps


def simulate_cascade(sg: SocialGraph, n_rounds: int = 10, seed_frac: float = 0.012,
                     seed: int = 0, coefs: dict | None = None):
    """Run the diffusion. Returns the round each node activated at (-1 = never)."""
    coefs = coefs or CASCADE_COEFS
    rng = np.random.default_rng(seed + 991)
    N = sg.n_nodes
    adj_sets = [set(a.tolist()) for a in sg.adj_list]
    # Sets, not arrays: the component count does membership tests, which are O(1) here.
    degree = np.array([len(a) for a in sg.adj_list], dtype=np.float32)
    log_deg = np.log1p(degree)

    active_at = np.full(N, -1, dtype=np.int32)

    seed_p = degree / degree.sum()
    n_seed = max(8, int(seed_frac * N))
    seeds = rng.choice(N, size=n_seed, replace=False, p=seed_p)
    # Degree-biased seeding. Real cascades usually start at hubs, and seeding uniformly
    # tends to produce cascades that die out immediately.
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
            # No active neighbour means no exposure, so skip. This also defines who
            # becomes a labelled instance later.
            div = _active_component_count(act_nbrs, adj_sets)
            h = (coefs["intercept"]
                 + coefs["n_active"] * np.log1p(n_a)
                 + coefs["frac_active"] * (n_a / max(nbrs.size, 1))
                 + coefs["diversity"] * np.log1p(div)
                 + coefs["susceptibility"] * sg.susceptibility[v]
                 + coefs["log_degree"] * log_deg[v])
            # log1p on the counts gives diminishing returns: going from 1 to 2 active
            # friends should matter more than going from 20 to 21.
            if rng.random() < 1.0 / (1.0 + np.exp(-h)):
                newly.append(v)
        if not newly:
            break
        active_at[np.array(newly, dtype=np.int32)] = t
        # Updated only after the loop, so everyone in round t sees the same world.
        # Updating inside the loop would let activation order leak into the outcome.

    return active_at


# ======================================================================================
# 3. Ego network sampling
# ======================================================================================

def _khop_pool(v: int, adj_list: list, k: int, cap: int, rng) -> np.ndarray:
    """BFS out to k hops, capped. Candidate pool with v at position 0."""
    seen = {v}
    order = [v]
    frontier = [v]
    for _ in range(k):
        nxt = []
        for u in frontier:
            nbrs = adj_list[u]
            if nbrs.size and len(seen) + nbrs.size > cap * 3:
                nbrs = rng.choice(nbrs, size=min(nbrs.size, 60), replace=False)
                # Subsample hubs. Without this, one celebrity neighbour drags in
                # thousands of nodes and the PPR step below crawls.
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
    # 3x the final size: enough candidates for the ranking below to be a real choice,
    # small enough that the dense PPR matrix stays cheap.


def _personalised_pagerank(pool: np.ndarray, adj_list: list, alpha: float = 0.15,
                           n_iter: int = 25) -> np.ndarray:
    """Personalised PageRank from the ego, restricted to the candidate pool."""
    # This is the closed-form version of random-walk-with-restart. The stationary
    # distribution IS the expected visit frequency of that walk, so we get the same
    # ranking DeepInf's sampler gets, deterministically and much faster.
    n = pool.size
    idx = {int(u): i for i, u in enumerate(pool)}
    P = np.zeros((n, n), dtype=np.float32)
    for i, u in enumerate(pool):
        nbrs = adj_list[u]
        local = [idx[int(w)] for w in nbrs if int(w) in idx]
        if local:
            P[i, local] = 1.0 / len(local)
            # Row-stochastic: uniform probability over neighbours inside the pool.
    e = np.zeros(n, dtype=np.float32)
    e[0] = 1.0                 # restart always returns to the ego, at index 0
    r = e.copy()
    for _ in range(n_iter):
        r = (1.0 - alpha) * (P.T @ r) + alpha * e
        # Power iteration. 25 steps is plenty: with alpha=0.15 the error shrinks by
        # 0.85 each step, so it is below 2% of the initial value by the end.
    return r


def sample_ego_network(v: int, adj_list: list, size: int, k_hops: int,
                       rng, strategy: str = "ppr"):
    """Pick `size` nodes around v and return (node_ids, local adjacency). Ego at index 0."""
    # Fixed size matters: it lets every instance be stored and batched as a dense
    # [B, s, s] tensor, which is why this project needs no sparse-graph library.
    pool = _khop_pool(v, adj_list, k_hops, cap=size, rng=rng)
    if pool.size > size:
        if strategy == "ppr":
            scores = _personalised_pagerank(pool, adj_list)
            scores[0] = np.inf
            # Force the ego to survive the top-k cut. Dropping the ego would be silent
            # and catastrophic, since the model reads its prediction off row 0.
            keep = np.argsort(-scores)[:size]
            keep = np.sort(keep)
            keep = np.concatenate([[0], keep[keep != 0]])[:size]
            # Re-sort so the ego lands back at index 0 after the argsort reshuffle.
        else:
            keep = np.arange(size)   # plain BFS order, kept for the sampling ablation
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
                # Induced subgraph: an edge is kept only when BOTH endpoints were
                # sampled. Edges leaving the ego network are dropped.
    np.fill_diagonal(adj, 0)
    # No self-loops here. The GNN layers add them back themselves (the +I in GCN).
    return nodes.astype(np.int32), adj


# ======================================================================================
# 4. Handcrafted features
# ======================================================================================

# Exactly the block the problem statement lists in section 2.4, nothing extra.
PS_HAND_FEATURES = [
    "log_degree", "clustering_coef", "log_n_active", "frac_active",
    "influence_ratio", "mean_active_nbr_logdeg", "max_active_nbr_logdeg",
    "attr_log_activity", "attr_tenure", "attr_hist_rate",
]

# Computed, stored, and deliberately kept out of the default feature set. A 2-hop active
# count is a hand-built proxy for the structural signal the GNN is supposed to learn, so
# giving it to the baseline for free answers a different question. It is reported as a
# separate strengthened baseline instead, and it turns out to change nothing.
EXTRA_HAND_FEATURES = ["log_active_2hop"]

HAND_FEATURE_NAMES = PS_HAND_FEATURES + EXTRA_HAND_FEATURES
N_PS_FEATURES = len(PS_HAND_FEATURES)


def handcrafted_features(v: int, sg: SocialGraph, active: np.ndarray,
                         clustering: np.ndarray) -> np.ndarray:
    """Ego-level features. Order must match HAND_FEATURE_NAMES."""
    nbrs = sg.adj_list[v]
    deg = max(nbrs.size, 1)      # max(.,1) so isolated nodes cannot divide by zero
    act_mask = active[nbrs]
    act_nbrs = nbrs[act_mask]
    n_a = act_nbrs.size

    nbr_deg = np.array([sg.adj_list[u].size for u in nbrs], dtype=np.float32)
    act_deg = nbr_deg[act_mask]
    influence_ratio = (act_deg.sum() / nbr_deg.sum()) if nbr_deg.sum() > 0 else 0.0
    # Degree share of the adopters. Catches "two well-connected friends adopted" as
    # distinct from "two isolated friends adopted", which a plain count cannot.

    two_hop_active = 0
    for u in nbrs:
        two_hop_active += int(active[sg.adj_list[u]].sum())
    # Counts with multiplicity on purpose, so a node reachable by several paths counts
    # several times. That redundancy is itself a weak signal about local structure.

    return np.array([
        np.log1p(deg),
        clustering[v],
        np.log1p(n_a),
        n_a / deg,
        influence_ratio,
        np.log1p(act_deg).mean() if n_a else 0.0,
        np.log1p(act_deg).max() if n_a else 0.0,
        # The n_a guards matter: .mean() on an empty array returns nan, and one nan
        # anywhere silently poisons the whole training run.
        sg.attrs[v, 0], sg.attrs[v, 1], sg.attrs[v, 2],
        np.log1p(two_hop_active),
    ], dtype=np.float32)


# ======================================================================================
# 5. Instance construction
# ======================================================================================

@dataclass
class InstanceSet:
    node_ids: np.ndarray     # [M, s] int32, ego at column 0
    adj: np.ndarray          # [M, s, s] uint8
    active: np.ndarray       # [M, s] uint8, action states at observation time
    valid: np.ndarray        # [M, s] uint8, padding mask for small ego networks
    hand: np.ndarray         # [M, H] float32
    y: np.ndarray            # [M] int8
    ego: np.ndarray          # [M] int32, needed for the group-wise split
    round_idx: np.ndarray    # [M] int16, which observation round this row came from
    hand_names: list


def build_instances(sg: SocialGraph, active_at: np.ndarray, observe_round: int,
                    horizon: int, ego_size: int = 50, k_hops: int = 2,
                    strategy: str = "ppr", max_instances: int | None = None,
                    seed: int = 0) -> InstanceSet:
    """Freeze the cascade at `observe_round` and label who adopts within `horizon`."""
    rng = np.random.default_rng(seed + 77)
    active = (active_at >= 0) & (active_at <= observe_round)
    # Everything the model sees comes from this snapshot. Anything after observe_round
    # is future information and must not touch the features.
    clustering = np.zeros(sg.n_nodes, dtype=np.float32)
    for node, c in nx.clustering(sg.G).items():
        clustering[node] = c
    # Computed once for the whole graph, not per instance. It is a global property and
    # recomputing it 12,000 times would dominate the runtime.

    candidates = []
    for v in range(sg.n_nodes):
        if active[v]:
            continue          # already adopted, so there is nothing left to predict
        nbrs = sg.adj_list[v]
        if nbrs.size == 0 or not active[nbrs].any():
            continue          # never exposed, so no influence question to ask
        candidates.append(v)
    candidates = np.array(candidates, dtype=np.int32)
    rng.shuffle(candidates)
    if max_instances is not None:
        candidates = candidates[:max_instances]
        # Shuffle before truncating. Node ids correlate with degree in a preferential
        # attachment graph, so taking the first N unshuffled would bias towards hubs.

    M, s = candidates.size, ego_size
    node_ids = np.zeros((M, s), dtype=np.int32)
    adjs = np.zeros((M, s, s), dtype=np.uint8)
    acts = np.zeros((M, s), dtype=np.uint8)
    valid = np.zeros((M, s), dtype=np.uint8)
    hand = np.zeros((M, len(HAND_FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(M, dtype=np.int8)
    # uint8 for adjacency rather than float32: at [12000, 50, 50] that is 30 MB instead
    # of 120 MB. The cast to float happens per batch.

    for i, v in enumerate(candidates):
        nodes, a = sample_ego_network(v, sg.adj_list, s, k_hops, rng, strategy)
        n = nodes.size
        node_ids[i, :n] = nodes
        adjs[i, :n, :n] = a
        acts[i, :n] = active[nodes].astype(np.uint8)
        acts[i, 0] = 0
        # Zero the ego's own action channel. It is 0 by construction already (we only
        # sampled inactive users), but this is the line that makes label leakage
        # impossible rather than merely unlikely.
        valid[i, :n] = 1
        hand[i] = handcrafted_features(v, sg, active, clustering)
        y[i] = 1 if (0 <= active_at[v] <= observe_round + horizon) else 0

    rounds = np.full(M, observe_round, dtype=np.int16)
    # Needed downstream: the oracle baseline has to score each row against the snapshot
    # it actually came from. Using one shared snapshot was a real bug in this project.
    return InstanceSet(node_ids, adjs, acts, valid, hand, y, candidates, rounds,
                       list(HAND_FEATURE_NAMES))


# ======================================================================================
# 6. Dataset builder with caching
# ======================================================================================

def concat_instances(parts: list[InstanceSet]) -> InstanceSet:
    """Stack instance sets from several observation rounds into one."""
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
    """Graph -> cascade -> instances at several observation times. Cached on disk."""
    # Several rounds rather than one, as DeepInf does. A single snapshot gives too few
    # labelled users and it samples only one phase of the cascade, and early adopters
    # genuinely look different from late ones.
    os.makedirs(cache_dir, exist_ok=True)
    rounds_tag = "-".join(map(str, observe_rounds))
    tag = (f"n{n_nodes}_s{ego_size}_k{k_hops}_{strategy}_o{rounds_tag}"
           f"_h{horizon}_{regime}_seed{seed}")
    # Every parameter that changes the data goes in the filename. Miss one and you get
    # a stale cache hit, which looks like a result rather than like an error.
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
        # Keep every snapshot. The oracle baseline needs to look up the right one per row.
    inst = concat_instances(parts)
    payload = (sg, active_at, inst, snaps)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


def split_indices(inst: InstanceSet, seed: int = 0, fracs=(0.6, 0.15, 0.25)):
    """Train/val/test split grouped by ego user."""
    # Grouping matters. A user appears once per observation round, so splitting rows at
    # random puts the same person in train and test with a nearly identical ego network
    # and inflates the test score. Peripheral nodes can still be shared across splits,
    # which is inherent to graph tasks and true of DeepInf too.
    rng = np.random.default_rng(seed + 4242)
    users = np.unique(inst.ego)
    perm = rng.permutation(users)
    a = int(fracs[0] * users.size)
    b = a + int(fracs[1] * users.size)
    groups = {"train": set(perm[:a].tolist()), "val": set(perm[a:b].tolist()),
              "test": set(perm[b:].tolist())}
    # Sets, not arrays: the membership test below runs once per instance.
    out = []
    for key in ("train", "val", "test"):
        g = groups[key]
        out.append(np.array([i for i, e in enumerate(inst.ego) if int(e) in g],
                            dtype=np.int64))
    return tuple(out)

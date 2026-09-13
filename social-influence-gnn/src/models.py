"""
models.py
---------
The encoder + fusion + head stack from section 2 of the problem statement.

Everything runs on dense per-instance adjacency tensors of shape [B, s, s]. Ego networks
are small and fixed-size, so dense is both simpler and faster here than a sparse
message-passing library, and it removes torch-geometric as a dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Instance normalisation  (section 2.3)
# --------------------------------------------------------------------------------------

class InstanceNorm(nn.Module):
    """
    Normalise each feature channel across the NODES of a single ego network.

        x'[b, i, c] = (x[b, i, c] - mu[b, c]) / (sigma[b, c] + eps)

    Read the axes carefully, because this is the part people get wrong in interviews:
    BatchNorm computes statistics across the batch; this computes them across the nodes
    *within one instance*, independently for every instance.

    Why it helps here: ego networks differ wildly in scale. A celebrity's neighbourhood
    has huge degree values and a long-tail user's has tiny ones. Without normalisation
    the network spends capacity learning "this is a big neighbourhood" instead of
    "this neighbourhood has a persuasive shape". Instance norm removes the per-instance
    location and scale, leaving the relative structure — which is the part that
    generalises. DeepInf reports it as a meaningful regulariser, and our ablation
    reproduces that.
    """

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, s, C]   mask: [B, s] (1 = real node, 0 = padding)
        m = mask.unsqueeze(-1).float()
        cnt = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu = (x * m).sum(dim=1, keepdim=True) / cnt
        var = (((x - mu) ** 2) * m).sum(dim=1, keepdim=True) / cnt
        out = (x - mu) / torch.sqrt(var + self.eps)
        if self.affine:
            out = out * self.gamma + self.beta
        return out * m


# --------------------------------------------------------------------------------------
# Graph encoders  (section 2.2)
# --------------------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """
    Kipf & Welling propagation:  H' = sigma( D^-1/2 (A + I) D^-1/2  H  W )

    The symmetric normalisation is what stops high-degree nodes from dominating the
    sum and blowing up activations; the +I keeps a node's own signal in the mix.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=True)

    def forward(self, h, adj_norm, mask):
        return self.lin(adj_norm @ h) * mask.unsqueeze(-1)


class GATLayer(nn.Module):
    """
    Velickovic et al. multi-head attention.

        e_ij   = LeakyReLU( a^T [ W h_i || W h_j ] )
        alpha  = softmax_j( e_ij ) over j in N(i) U {i}
        h'_i   = || _heads  sigma( sum_j alpha_ij W h_j )

    Why attention rather than GCN for this task: influence is not uniform. One close
    friend can outweigh twenty acquaintances, and a fixed degree-normalised average
    cannot express that. The attention weights are also the one part of the model you
    can actually show a stakeholder — "this adoption was driven by these two people".
    """

    def __init__(self, d_in: int, d_out: int, heads: int = 4, dropout: float = 0.2,
                 concat: bool = True):
        super().__init__()
        self.heads, self.d_out, self.concat = heads, d_out, concat
        self.W = nn.Linear(d_in, heads * d_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, 1, d_out))
        self.a_dst = nn.Parameter(torch.empty(1, heads, 1, d_out))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.dropout = dropout

    def forward(self, h, adj, mask):
        B, s, _ = h.shape
        Wh = self.W(h).view(B, s, self.heads, self.d_out).permute(0, 2, 1, 3)  # [B,H,s,d]

        # Decomposing a^T[x||y] into a_src^T x + a_dst^T y turns an O(s^2 d) gather
        # into two O(s d) projections plus a broadcast add. Standard GAT trick.
        e_src = (Wh * self.a_src).sum(-1).unsqueeze(-1)   # [B,H,s,1]
        e_dst = (Wh * self.a_dst).sum(-1).unsqueeze(-2)   # [B,H,1,s]
        e = F.leaky_relu(e_src + e_dst, negative_slope=0.2)

        pair_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()
        eye = torch.eye(s, device=h.device, dtype=torch.bool).unsqueeze(0)
        allowed = ((adj.bool() | eye) & pair_mask).unsqueeze(1)   # [B,1,s,s]

        e = e.masked_fill(~allowed, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)          # fully padded rows
        alpha = F.dropout(alpha, self.dropout, self.training)

        out = alpha @ Wh                                   # [B,H,s,d]
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, s, -1) if self.concat else out.mean(dim=2)
        return out * mask.unsqueeze(-1)


def normalise_adj(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2, respecting padding."""
    B, s, _ = adj.shape
    eye = torch.eye(s, device=adj.device).unsqueeze(0)
    a = adj.float() + eye
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    a = a * pair_mask
    deg = a.sum(-1).clamp(min=1e-6)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(-1) * a * dinv.unsqueeze(-2)


# --------------------------------------------------------------------------------------
# Full model  (sections 2.2 - 2.5)
# --------------------------------------------------------------------------------------

class InfluenceModel(nn.Module):
    """
    node features -> input projection -> [instance norm] -> L x {GCN|GAT}
                  -> take ego row -> [|| handcrafted features] -> MLP -> sigmoid

    Every component the ablation touches is a constructor flag, so the ablation table
    is produced by the same code path as the main model. No second implementation to
    drift out of sync.
    """

    def __init__(self, d_node: int, d_hand: int, hidden: int = 64, layers: int = 2,
                 encoder: str = "gat", heads: int = 4, dropout: float = 0.3,
                 use_instance_norm: bool = True, use_hand: bool = True,
                 use_struct_emb: bool = True, d_struct: int = 0,
                 use_wide: bool = True):
        super().__init__()
        self.encoder_type = encoder
        self.use_instance_norm = use_instance_norm
        self.use_hand = use_hand
        self.use_struct_emb = use_struct_emb

        d_in = d_node + (d_struct if use_struct_emb else 0)
        self.in_proj = nn.Linear(d_in, hidden)
        self.inorm = InstanceNorm(hidden) if use_instance_norm else None

        self.layers = nn.ModuleList()
        d = hidden
        for i in range(layers):
            last = (i == layers - 1)
            if encoder == "gcn":
                self.layers.append(GCNLayer(d, hidden))
                d = hidden
            else:
                self.layers.append(GATLayer(d, hidden // heads, heads=heads,
                                            dropout=dropout, concat=True))
                d = (hidden // heads) * heads
        self.dropout = dropout

        d_head = d + (d_hand if use_hand else 0)
        self.head = nn.Sequential(
            nn.Linear(d_head, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Wide-and-deep skip (Cheng et al., 2016).
        #
        # Without it, the 10 clean handcrafted features are concatenated to a 64-dim
        # noisy GNN embedding and then squeezed through a dropout MLP - they get
        # diluted, and the fused model scores BELOW plain logistic regression. That is
        # a real result we measured, not a hypothetical.
        #
        # The skip gives the handcrafted block a direct linear path to the logit, so
        # the architecture is literally "logistic regression + a learned graph
        # correction". The floor becomes the baseline instead of something the
        # optimiser has to rediscover, and the GNN branch only has to carry the
        # residual structural signal.
        self.wide = nn.Linear(d_hand, 1) if (use_hand and use_wide) else None

    def forward(self, x_node, adj, mask, hand, struct=None, return_attn=False):
        if self.use_struct_emb and struct is not None:
            x_node = torch.cat([x_node, struct], dim=-1)

        h = self.in_proj(x_node) * mask.unsqueeze(-1)
        if self.inorm is not None:
            h = self.inorm(h, mask)

        adj_norm = normalise_adj(adj, mask) if self.encoder_type == "gcn" else None
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(h, adj_norm, mask) if self.encoder_type == "gcn" else layer(h, adj, mask)
            h = F.elu(h)
            if h.shape == h_in.shape and i > 0:
                h = h + h_in                                  # residual, helps depth>2
            h = F.dropout(h, self.dropout, self.training)

        h_ego = h[:, 0, :]                                    # ego sits at index 0
        z = torch.cat([h_ego, hand], dim=-1) if self.use_hand else h_ego
        logit = self.head(z).squeeze(-1)
        if self.wide is not None:
            logit = logit + self.wide(hand).squeeze(-1)
        return logit

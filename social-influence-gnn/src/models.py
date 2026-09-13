"""
GNN encoders and the prediction head.

Everything runs on dense per-instance adjacency [B, s, s]. Ego networks are small and
fixed-size, so dense is simpler and faster here than a sparse message-passing library,
and it keeps torch-geometric out of the dependency list.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# Instance normalisation
# ======================================================================================

class InstanceNorm(nn.Module):
    """Normalise each feature channel across the NODES of one ego network."""
    # Worth getting the axis straight, because it is the usual interview trip-up:
    #   BatchNorm    -> statistics across the batch
    #   LayerNorm    -> across the channels of one example
    #   InstanceNorm -> across the nodes inside one ego network   (this one)
    #
    # Why it helps: a celebrity's neighbourhood has huge degree values and a lurker's has
    # tiny ones. Without this the model burns capacity learning "big neighbourhood"
    # instead of "persuasive shape". Removing it costs about 0.010 AUC in our ablation.

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))
            # Learnable scale and shift, so the layer can undo itself if that helps.

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, s, C], mask: [B, s] with 1 = real node, 0 = padding
        m = mask.unsqueeze(-1).float()
        cnt = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Divide by the real node count, not s. Padded rows would drag the mean towards
        # zero and the amount of drag would depend on how small the ego network was.
        mu = (x * m).sum(dim=1, keepdim=True) / cnt
        var = (((x - mu) ** 2) * m).sum(dim=1, keepdim=True) / cnt
        out = (x - mu) / torch.sqrt(var + self.eps)
        # eps guards the case where every node has the same value and var is 0.
        if self.affine:
            out = out * self.gamma + self.beta
        return out * m
        # Re-zero the padding. The affine shift just wrote beta into padded rows.


# ======================================================================================
# Graph encoders
# ======================================================================================

class GCNLayer(nn.Module):
    """One Kipf-Welling propagation step: H' = D^-1/2 (A+I) D^-1/2 H W."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=True)

    def forward(self, h, adj_norm, mask):
        return self.lin(adj_norm @ h) * mask.unsqueeze(-1)
        # Aggregate first, then transform. Doing it the other way costs more compute
        # when d_in > d_out, and gives the same result.


class GATLayer(nn.Module):
    """Multi-head graph attention (Velickovic et al.)."""
    # GCN weights every neighbour by degree alone. Attention learns the weight instead,
    # which suits influence: one close friend can outweigh twenty acquaintances, and a
    # degree-normalised average has no way to say that. The attention weights are also
    # the one part you can show a person: "these two neighbours drove this prediction".

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
        # Xavier keeps activation variance stable through the layer. Default init on the
        # attention vectors tends to saturate the softmax early in training.
        self.dropout = dropout

    def forward(self, h, adj, mask):
        B, s, _ = h.shape
        Wh = self.W(h).view(B, s, self.heads, self.d_out).permute(0, 2, 1, 3)  # [B,H,s,d]

        e_src = (Wh * self.a_src).sum(-1).unsqueeze(-1)   # [B,H,s,1]
        e_dst = (Wh * self.a_dst).sum(-1).unsqueeze(-2)   # [B,H,1,s]
        e = F.leaky_relu(e_src + e_dst, negative_slope=0.2)
        # The attention score is a^T[Wh_i || Wh_j]. Splitting a into a_src and a_dst
        # turns that concatenation into two small projections plus a broadcast add, so
        # we never build the [B,H,s,s,2d] concatenated tensor. Same numbers, far less
        # memory. LeakyReLU(0.2) is what the paper uses.

        pair_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()
        eye = torch.eye(s, device=h.device, dtype=torch.bool).unsqueeze(0)
        allowed = ((adj.bool() | eye) & pair_mask).unsqueeze(1)   # [B,1,s,s]
        # Attend to real neighbours plus yourself. The eye term is the self-loop the
        # sampler stripped out; without it a node cannot keep its own representation.

        e = e.masked_fill(~allowed, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        # -inf before softmax, not 0 after. Zeroing afterwards would leave the weights
        # unnormalised and every row summing to something different.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        # A fully padded row is all -inf, and softmax of all -inf is nan.
        alpha = F.dropout(alpha, self.dropout, self.training)
        # Dropout on the attention weights themselves, so the model cannot lean on one
        # neighbour every time.

        out = alpha @ Wh                                   # [B,H,s,d]
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, s, -1) if self.concat else out.mean(dim=2)
        # Concatenate heads in hidden layers, average them on the output layer.
        return out * mask.unsqueeze(-1)


def normalise_adj(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation D^-1/2 (A+I) D^-1/2, padding-aware."""
    B, s, _ = adj.shape
    eye = torch.eye(s, device=adj.device).unsqueeze(0)
    a = adj.float() + eye
    # +I is the self-loop: without it a node's own features vanish after one layer.
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    a = a * pair_mask
    deg = a.sum(-1).clamp(min=1e-6)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(-1) * a * dinv.unsqueeze(-2)
    # The two broadcasts scale rows and columns separately, which is the cheap way to
    # write D^-1/2 A D^-1/2 without building diagonal matrices. Symmetric rather than
    # row-normalised so high-degree neighbours do not dominate every sum they appear in.


# ======================================================================================
# Full model
# ======================================================================================

class InfluenceModel(nn.Module):
    """node features -> GNN -> ego embedding + handcrafted features -> logit."""
    # Every ablation is a constructor flag, so the ablation table runs through the same
    # code path as the main model and there is no second implementation to drift.

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
                # Split the hidden width across heads so GAT and GCN have a comparable
                # parameter count. Otherwise a GAT win could just be extra capacity.
        self.dropout = dropout

        d_head = d + (d_hand if use_hand else 0)
        self.head = nn.Sequential(
            nn.Linear(d_head, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        self.wide = nn.Linear(d_hand, 1) if (use_hand and use_wide) else None
        # Wide-and-deep skip (Cheng et al. 2016), and it is load-bearing.
        # Without it, 10 clean handcrafted features get concatenated onto a 64-dim noisy
        # GNN embedding and squeezed through a dropout MLP. They get diluted, and the
        # fused model scored BELOW plain logistic regression (0.6532 vs 0.6616). That is
        # measured, not hypothetical.
        # With it, the architecture is "logistic regression plus a learned graph
        # correction", so the baseline becomes the floor instead of something the
        # optimiser has to rediscover.

    def forward(self, x_node, adj, mask, hand, struct=None, return_attn=False):
        if self.use_struct_emb and struct is not None:
            x_node = torch.cat([x_node, struct], dim=-1)

        h = self.in_proj(x_node) * mask.unsqueeze(-1)
        if self.inorm is not None:
            h = self.inorm(h, mask)
            # After the projection, before message passing, which is where DeepInf puts it.

        adj_norm = normalise_adj(adj, mask) if self.encoder_type == "gcn" else None
        # GCN needs the normalised adjacency, GAT works off the raw one because it
        # computes its own weights. Building it once outside the loop.
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(h, adj_norm, mask) if self.encoder_type == "gcn" else layer(h, adj, mask)
            h = F.elu(h)
            if h.shape == h_in.shape and i > 0:
                h = h + h_in
                # Residual from layer 2 onwards. Repeated neighbourhood averaging pulls
                # every node towards the same vector (over-smoothing), and the skip gives
                # the node's own signal a path forward. Our 3-layer ablation still loses
                # 0.018 AUC, so it softens the problem rather than solving it.
            h = F.dropout(h, self.dropout, self.training)

        h_ego = h[:, 0, :]
        # Row 0 is the ego. This is why the sampler is careful to keep it there.
        z = torch.cat([h_ego, hand], dim=-1) if self.use_hand else h_ego
        logit = self.head(z).squeeze(-1)
        if self.wide is not None:
            logit = logit + self.wide(hand).squeeze(-1)
        return logit
        # Raw logits, no sigmoid. BCEWithLogitsLoss applies it internally in a
        # numerically stable way, and applying it twice would quietly break training.

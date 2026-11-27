"""Point Transformer building blocks (Zhao et al. 2021, arXiv 2012.09164) in pure PyTorch.

Shared by the target encoder (self-attention over the target cloud) and the matcher (self-attention among touch points,
cross-attention from target tokens to touch tokens). LayerNorm replaces BatchNorm because batches come from RL rollouts.
"""

import torch
from torch import Tensor, nn

from tacfind.networks import mlp


def gather(x: Tensor, idx: Tensor) -> Tensor:
    """Gather rows per batch element: x (B, N, ...) and idx (B, *S) long -> (B, *S, ...)."""
    batch = torch.arange(x.shape[0], device=x.device).view(-1, *[1] * (idx.dim() - 1))
    return x[batch, idx]


@torch.no_grad()
def knn(pos_q: Tensor, pos_kv: Tensor, k: int, kv_mask: Tensor | None = None) -> Tensor:
    """Indices of the k nearest keys of every query; masked keys are never chosen while enough valid ones exist.

    Args:
        pos_q: (B, Q, 3) query positions.
        pos_kv: (B, N, 3) key positions.
        k: neighbours per query (clipped to N).
        kv_mask: (B, N) bool, True = real point.
    Returns:
        (B, Q, k) long. When fewer than k keys are valid the surplus indices point at masked keys, so callers must still
        apply `kv_mask` to the gathered neighbours (`VectorAttention` does).
    """
    dist = torch.cdist(pos_q, pos_kv)  # (B, Q, N)
    if kv_mask is not None:
        dist = dist.masked_fill(~kv_mask[:, None], float("inf"))
    return dist.topk(min(k, dist.shape[-1]), largest=False).indices


@torch.no_grad()
def farthest_point_sample(pos: Tensor, m: int, mask: Tensor | None = None, start: Tensor | None = None) -> Tensor:
    """Farthest point sampling (deterministic).

    Args:
        pos: (B, N, 3) positions.
        m: number of points to keep (<= N).
        mask: (B, N) bool, True = real point. Padded points are picked only once every real point is taken.
        start: (B,) long index of the first point; default 0.
    Returns:
        (B, m) long indices, all distinct.
    """
    B, N, _ = pos.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=pos.device)
    # Distance to the chosen set for unchosen real points (>= 0); padded = -1 and chosen = -2 rank below them.
    dist = torch.full((B, N), float("inf"), dtype=pos.dtype, device=pos.device)
    if mask is not None:
        dist = dist.masked_fill(~mask, -1.0)
    far = start if start is not None else torch.zeros(B, dtype=torch.long, device=pos.device)
    for i in range(m):
        idx[:, i] = far
        d = (pos - gather(pos, far[:, None])).square().sum(-1)
        dist = torch.where(dist >= 0, torch.minimum(dist, d), dist).scatter(1, far[:, None], -2.0)
        far = dist.argmax(-1)
    return idx


class VectorAttention(nn.Module):
    """Grouped vector attention (Point Transformer; grouped as in Point Transformer V2, arXiv 2210.05666).

    y_i = sum_j w_ij * (alpha(x_j) + delta_ij),  w_ij = softmax_j(gamma(phi(q_i) - psi(k_j) + delta'_ij)),
    with the subtraction relation in a reduced width R and one weight per group of C / G channels (softmax over the
    neighbours j). delta_ij / delta'_ij are linear read-outs of a shared position encoding h_ij = GELU(W (p_i - p_j));
    without positions both are 0. groups = rel_dim = dim gives the original per-channel layer of Zhao et al. 2021.
    Memory: per (query, neighbour) pair only R- and G-wide tensors exist; the C-wide value term is never built per
    pair for full attention, and the positional value term is applied after aggregation (it is linear in h_ij).
    """

    def __init__(self, dim: int, groups: int | None = None, rel_dim: int | None = None):
        """Args: dim: feature width C. groups: weight groups G (divides C; default C). rel_dim: relation width R."""
        super().__init__()
        self.groups, rel_dim = groups or dim, rel_dim or dim
        assert dim % self.groups == 0, (dim, self.groups)
        self.phi, self.psi, self.alpha = nn.Linear(dim, rel_dim), nn.Linear(dim, rel_dim), nn.Linear(dim, dim)
        self.theta = nn.Sequential(nn.Linear(3, rel_dim), nn.GELU())  # shared position encoding h_ij
        self.theta_rel, self.theta_val = nn.Linear(rel_dim, rel_dim), nn.Linear(rel_dim, dim)
        self.gamma = mlp(rel_dim, rel_dim, self.groups)

    def forward(self, x_q: Tensor, x_kv: Tensor, pos_q: Tensor | None = None, pos_kv: Tensor | None = None,
                idx: Tensor | None = None, kv_mask: Tensor | None = None) -> Tensor:
        """Attend from queries to their neighbour keys.

        Args:
            x_q: (B, Q, C) query features.
            x_kv: (B, N, C) key/value features.
            pos_q, pos_kv: (B, Q, 3), (B, N, 3) positions in one shared frame, or None (no positional term).
            idx: (B, Q, K) neighbour indices into the keys, or None = every key is a neighbour.
            kv_mask: (B, N) bool, True = real key. A query without any valid neighbour gets zeros.
        Returns:
            (B, Q, C).
        """
        nb = (lambda t: t[:, None]) if idx is None else (lambda t: gather(t, idx))  # keys -> (B, Q|1, K, ...)
        rel = self.phi(x_q)[:, :, None] - nb(self.psi(x_kv))  # (B, Q, K, R)
        if pos_q is not None:
            h = self.theta(pos_q[:, :, None] - nb(pos_kv))  # (B, Q, K, R)
            rel = rel + self.theta_rel(h)
        logits = self.gamma(rel)  # (B, Q, K, G)
        if kv_mask is None:
            w = logits.softmax(2)
        else:  # a finite fill (not -inf) keeps all-masked rows NaN-free; multiplying by `valid` then zeroes them
            valid = nb(kv_mask)[..., None]
            w = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(2) * valid
        v = self.alpha(x_kv).unflatten(-1, (self.groups, -1))  # (B, N, G, C/G)
        if idx is None:  # values do not depend on the query: no per-pair copy
            y = torch.einsum("bqkg,bkgc->bqgc", w, v)
        else:
            y = torch.einsum("bqkg,bqkgc->bqgc", w, gather(v, idx))
        if pos_q is not None:  # sum_j w_ij (W h_ij + b) = W_g (sum_j w_ij h_ij) + b * sum_j w_ij, per group g
            weight = self.theta_val.weight.unflatten(0, (self.groups, -1))  # (G, C/G, R)
            y = y + torch.einsum("bqgr,gcr->bqgc", torch.einsum("bqkg,bqkr->bqgr", w, h), weight)
            y = y + self.theta_val.bias.unflatten(0, (self.groups, -1)) * w.sum(2)[..., None]
        return y.flatten(2)


class PointTransformerBlock(nn.Module):
    """Residual block: LayerNorm -> linear -> vector self-attention over kNN -> linear, plus skip."""

    def __init__(self, dim: int, k: int = 16, groups: int | None = None, rel_dim: int | None = None):
        """Args: dim: width. k: neighbours (all points when the cloud has at most k). groups, rel_dim: attention."""
        super().__init__()
        self.k = k
        self.norm, self.lin_in, self.lin_out = nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn = VectorAttention(dim, groups, rel_dim)

    def forward(self, x: Tensor, pos: Tensor, mask: Tensor | None = None) -> Tensor:
        """Args: x (B, N, C), pos (B, N, 3), mask (B, N) bool or None. Returns: (B, N, C)."""
        idx = knn(pos, pos, self.k, mask) if self.k < x.shape[1] else None
        h = self.lin_in(self.norm(x))
        return x + self.lin_out(self.attn(h, h, pos, pos, idx, mask))


class TransitionDown(nn.Module):
    """FPS to m centres -> kNN group -> MLP on [relative position, feature] -> max-pool per group."""

    def __init__(self, dim: int, m: int, k: int = 16):
        """Args: dim: feature width (in = out). m: points kept. k: group size."""
        super().__init__()
        self.m, self.k = m, k
        self.mlp = nn.Sequential(nn.Linear(dim + 3, dim), nn.LayerNorm(dim), nn.GELU())

    def forward(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Args: x (B, N, C), pos (B, N, 3) of a dense cloud. Returns: features (B, m, C), positions (B, m, 3)."""
        pos_c = gather(pos, farthest_point_sample(pos, self.m))  # (B, m, 3)
        idx = knn(pos_c, pos, self.k)  # (B, m, k)
        group = torch.cat([gather(pos, idx) - pos_c[:, :, None], gather(x, idx)], -1)
        return self.mlp(group).max(2).values, pos_c

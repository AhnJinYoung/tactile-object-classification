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
def farthest_point_sample(pos: Tensor, m: int) -> Tensor:
    """Farthest point sampling, starting from point 0 (deterministic).

    Args:
        pos: (B, N, 3) positions.
        m: number of points to keep.
    Returns:
        (B, m) long indices.
    """
    B, N, _ = pos.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=pos.device)
    dist = torch.full((B, N), float("inf"), device=pos.device)
    far = torch.zeros(B, dtype=torch.long, device=pos.device)
    for i in range(m):
        idx[:, i] = far
        dist = torch.minimum(dist, (pos - gather(pos, far[:, None])).square().sum(-1))
        far = dist.argmax(-1)
    return idx


class VectorAttention(nn.Module):
    """Vector attention: y_i = sum_j softmax_j(gamma(phi(q_i) - psi(k_j) + delta_ij)) * (alpha(k_j) + delta_ij).

    The softmax is per channel over the neighbours j; delta_ij = theta(p_i - p_j), or 0 without positions. Serves both
    self-attention (queries = keys, kNN neighbours) and cross-attention (all keys, no positions).
    """

    def __init__(self, dim: int):
        """Args: dim: feature width C of queries, keys and output."""
        super().__init__()
        self.phi, self.psi, self.alpha = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.theta, self.gamma = mlp(3, dim, dim), mlp(dim, dim, dim)

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
        q = self.phi(x_q)[:, :, None]  # (B, Q, 1, C)
        k, v = nb(self.psi(x_kv)), nb(self.alpha(x_kv))
        delta = self.theta(pos_q[:, :, None] - nb(pos_kv)) if pos_q is not None else 0.0
        logits = self.gamma(q - k + delta)  # (B, Q, K, C)
        if kv_mask is None:
            return (logits.softmax(2) * (v + delta)).sum(2)
        # A finite fill (not -inf) keeps all-masked rows NaN-free; multiplying by `valid` then zeroes them.
        valid = nb(kv_mask)[..., None]
        w = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(2) * valid
        return (w * (v + delta)).sum(2)


class PointTransformerBlock(nn.Module):
    """Residual block: LayerNorm -> linear -> vector self-attention over kNN -> linear, plus skip."""

    def __init__(self, dim: int, k: int = 16):
        """Args: dim: feature width. k: neighbours per point (all points when the cloud has at most k)."""
        super().__init__()
        self.k = k
        self.norm, self.lin_in, self.lin_out = nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn = VectorAttention(dim)

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

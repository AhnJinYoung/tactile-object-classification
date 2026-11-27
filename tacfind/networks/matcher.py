"""Point Transformer cross-modal matcher: compares the touch evidence with the target object.

Two sub-modules, trained in different stages:
- `embed`: touch side -> tokens + pooled contrastive embedding e_touch. The touch-side input is pluggable (config group
  `matcher`): `PointTokenizer` (tactile contact points), `LatentTokenizer` (z_touch split into tokens) or
  `LatentHeadTokenizer` (z_touch read as an estimated contact point cloud).
- `confidence`: vector cross-attention target tokens -> touch tokens (no positional term: the two clouds live in
  different frames) -> difference features pooled into z_diff -> confidence logit.
"""

import torch
from torch import Tensor, nn

from tacfind.networks import masked_mean, mlp
from tacfind.networks.point_transformer import PointTransformerBlock, VectorAttention, farthest_point_sample, gather
from tacfind.networks.target_encoder import TargetCode

Tokens = tuple[Tensor, Tensor, Tensor | None]  # tokens (B, T, dim), mask (B, T) bool, est_points (B, K, 3) | None


class PointTokenizer(nn.Module):
    """Tokens from tactile contact points: [centred xyz, normal, log1p(force), age] -> Point Transformer blocks.

    At most `max_tokens` valid points are kept, by farthest point sampling seeded at the most recent contact: 20 Hz
    contacts are highly redundant, and this keeps both geometric coverage and recency.
    """

    def __init__(self, dim: int, depth: int, k: int = 16, pos_scale: float = 0.1, max_tokens: int = 32,
                 groups: int | None = None, rel_dim: int | None = None):
        """Args:
            dim: width. depth: self-attention blocks. k: kNN size. pos_scale: m per network coordinate unit.
            max_tokens: cap on touch tokens. groups, rel_dim: grouped vector attention (see `VectorAttention`).
        """
        super().__init__()
        self.pos_scale, self.max_tokens = pos_scale, max_tokens
        self.embed = mlp(8, dim, dim)
        self.blocks = nn.ModuleList(PointTransformerBlock(dim, k, groups, rel_dim) for _ in range(depth))

    def forward(self, z_touch: Tensor, touch_pts: Tensor, touch_mask: Tensor) -> Tokens:
        # The one host sync of the forward pass: fewer tokens when no row has max_tokens contacts (the common case).
        n_keep = min(self.max_tokens, max(int(touch_mask.sum(1).max()), 1))
        recent = touch_pts[..., 7].masked_fill(~touch_mask, float("inf")).argmin(1)  # (B,) most recent contact
        keep = farthest_point_sample(touch_pts[..., :3], n_keep, touch_mask, recent)  # valid points come first
        pts, mask = gather(touch_pts, keep), gather(touch_mask, keep)
        xyz = pts[..., :3]
        pos = (xyz - masked_mean(xyz, mask)[:, None]) / self.pos_scale
        x = self.embed(torch.cat([pos, pts[..., 3:6], pts[..., 6:7].clamp(min=0).log1p(), pts[..., 7:8]], -1))
        for block in self.blocks:
            x = block(x, pos, mask)
        return x, mask, None


class LatentTokenizer(nn.Module):
    """z_touch split by a linear layer into a few tokens without positions."""

    def __init__(self, dim: int, n_tokens: int = 8):
        """Args: dim: width of z_touch and of the tokens. n_tokens: number of tokens."""
        super().__init__()
        self.n_tokens = n_tokens
        self.split = nn.Linear(dim, n_tokens * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_touch: Tensor, touch_pts: Tensor, touch_mask: Tensor) -> Tokens:
        x = self.norm(self.split(z_touch).unflatten(-1, (self.n_tokens, -1)))
        return x, torch.ones(x.shape[:2], dtype=torch.bool, device=x.device), None


class LatentHeadTokenizer(nn.Module):
    """First 3K dims of z_touch = K estimated contact points; the rest gives their features; then self-attention."""

    def __init__(self, dim: int, depth: int, n_points: int = 8, pos_scale: float = 0.1, groups: int | None = None,
                 rel_dim: int | None = None):
        """Args:
            dim: width (> 3 * n_points). depth: self-attention blocks. n_points: K. pos_scale: m per unit.
            groups, rel_dim: grouped vector attention (see `VectorAttention`).
        """
        super().__init__()
        self.n_points, self.pos_scale = n_points, pos_scale
        self.feat = nn.Linear(dim - 3 * n_points, n_points * dim)
        self.embed = mlp(3, dim, dim)
        self.blocks = nn.ModuleList(PointTransformerBlock(dim, n_points, groups, rel_dim) for _ in range(depth))

    def forward(self, z_touch: Tensor, touch_pts: Tensor, touch_mask: Tensor) -> Tokens:
        K = self.n_points
        pos = z_touch[:, : 3 * K].unflatten(-1, (K, 3))  # (B, K, 3), network units
        pos = pos - pos.mean(1, keepdim=True)  # only relative geometry is meaningful (Chamfer loss centres too)
        x = self.feat(z_touch[:, 3 * K:]).unflatten(-1, (K, -1)) + self.embed(pos)
        for block in self.blocks:
            x = block(x, pos)
        return x, torch.ones(x.shape[:2], dtype=torch.bool, device=x.device), pos * self.pos_scale


class TouchEmbed(nn.Module):
    """`matcher.embed`: a tokenizer + masked mean pooling -> e_touch (contrastive embedding of what is touched)."""

    def __init__(self, tokenizer: nn.Module, dim: int):
        super().__init__()
        self.tokenizer, self.head = tokenizer, nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, z_touch: Tensor, touch_pts: Tensor,
                touch_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Returns: tokens (B, T, dim), token mask (B, T), e_touch (B, dim), est_points (B, K, 3) or None."""
        tokens, mask, est_points = self.tokenizer(z_touch, touch_pts, touch_mask)
        return tokens, mask, self.head(masked_mean(tokens, mask)), est_points


class Confidence(nn.Module):
    """`matcher.confidence`: queries = target tokens, keys/values = touch tokens -> z_diff (B, dim), logit (B,)."""

    def __init__(self, dim: int, groups: int | None = None, rel_dim: int | None = None):
        """Args: dim: width. groups, rel_dim: grouped vector attention (see `VectorAttention`)."""
        super().__init__()
        self.norm_target, self.norm_touch = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = VectorAttention(dim, groups, rel_dim)
        self.diff = mlp(3 * dim, dim, dim, norm=True)
        self.head = nn.Linear(dim, 1)

    def forward(self, target: TargetCode, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        t = self.norm_target(target.tokens)  # (B, M, dim)
        explained = self.cross(t, self.norm_touch(tokens), kv_mask=mask)  # what touch explains of each target token
        z_diff = self.diff(torch.cat([t, explained, t - explained], -1)).mean(1)
        return z_diff, self.head(z_diff).squeeze(-1)


class CrossModalMatcher(nn.Module):
    """Touch evidence vs. target -> z_diff, confidence logit, e_touch, est_points."""

    def __init__(self, tokenizer: nn.Module, dim: int, groups: int | None = None, rel_dim: int | None = None):
        """Args:
            tokenizer: touch-side tokenizer (instantiated by Hydra from the `matcher` group). dim: width.
            groups, rel_dim: grouped vector attention of the cross-attention (see `VectorAttention`).
        """
        super().__init__()
        self.embed, self.confidence = TouchEmbed(tokenizer, dim), Confidence(dim, groups, rel_dim)

    def forward(self, z_touch: Tensor, touch_pts: Tensor, touch_mask: Tensor, target: TargetCode) -> dict:
        """Args:
            z_touch: (B, dim) from the touch encoder.
            touch_pts: (B, L, 8) = [pos_base xyz (m), normal (unit), force (N), age in [0, 1]].
            touch_mask: (B, L) bool, True = real contact.
            target: TargetCode with B rows.
        Returns:
            dict with z_diff (B, dim), conf_logit (B,), e_touch (B, dim), est_points (B, K, 3) in m or None.
        """
        tokens, mask, e_touch, est_points = self.embed(z_touch, touch_pts, touch_mask)
        z_diff, conf_logit = self.confidence(target, tokens, mask)
        return dict(z_diff=z_diff, conf_logit=conf_logit, e_touch=e_touch, est_points=est_points)

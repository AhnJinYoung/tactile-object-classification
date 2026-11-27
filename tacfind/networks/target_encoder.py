"""Target encoder: the target object's point cloud -> target tokens + a global embedding z_target.

The tokens are the queries of the matcher's cross-attention; z_target is the object's contrastive embedding. The
trainer encodes the whole object library once and gathers rows per sample (`TargetCode.__getitem__`).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tacfind.networks import mlp
from tacfind.networks.point_transformer import PointTransformerBlock, TransitionDown


@dataclass
class TargetCode:
    """Encoded target object.

    Attributes:
        tokens: (B, M, dim) token features.
        pos: (B, M, 3) token positions, object frame centred on the cloud centroid, m.
        z: (B, dim) global embedding (contrastive embedding of the object).
    """

    tokens: Tensor
    pos: Tensor
    z: Tensor

    def __getitem__(self, idx) -> "TargetCode":
        return TargetCode(self.tokens[idx], self.pos[idx], self.z[idx])


class TargetEncoder(nn.Module):
    """Point embedding -> per stage: transition-down + `depth` Point Transformer blocks -> max-pool -> linear."""

    def __init__(self, dim: int, depth: int, stages: Sequence[int] = (128, 64), k: int = 16, pos_scale: float = 0.1):
        """Args:
            dim: token width.
            depth: Point Transformer blocks per stage.
            stages: points kept by each transition-down; the last entry is M, the number of target tokens.
            k: kNN neighbourhood size.
            pos_scale: length (m) that maps to one unit of the network's coordinates.
        """
        super().__init__()
        self.pos_scale = pos_scale
        self.embed = mlp(6, dim, dim)
        self.down = nn.ModuleList(TransitionDown(dim, m, k) for m in stages)
        self.blocks = nn.ModuleList(nn.ModuleList(PointTransformerBlock(dim, k) for _ in range(depth)) for _ in stages)
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, dim)

    def forward(self, cloud: Tensor) -> TargetCode:
        """Args: cloud (B, N, 6) = [xyz in the object frame (m), unit outward normal]. Returns: TargetCode."""
        xyz = cloud[..., :3]
        pos = (xyz - xyz.mean(1, keepdim=True)) / self.pos_scale  # centred: invariant to where the frame origin is
        x = self.embed(torch.cat([pos, cloud[..., 3:6]], -1))
        for down, blocks in zip(self.down, self.blocks):
            x, pos = down(x, pos)
            for block in blocks:
                x = block(x, pos)
        x = self.norm(x)
        return TargetCode(x, pos * self.pos_scale, self.head(x.max(1).values))

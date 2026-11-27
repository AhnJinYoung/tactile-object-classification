"""Spatial-temporal encoders: observation history o_{t-H+1..t} -> z_touch.

Three interchangeable classes with the same call `forward(obs_hist, hist_mask) -> (B, dim)`; the config group
`touch_encoder` picks one. Rows of `obs_hist` before the episode start are zero and False in `hist_mask`.
"""

import torch
from torch import Tensor, nn

from tacfind.networks import masked_mean, mlp


class MLPEncoder(nn.Module):
    """Vanilla MLP over the flattened window (plus the mask, so padded zeros differ from real zeros)."""

    def __init__(self, obs_dim: int, history: int, dim: int, depth: int):
        """Args: obs_dim: D_o. history: H. dim: output width. depth: hidden layers of width 2 * dim."""
        super().__init__()
        self.net = mlp(history * (obs_dim + 1), *[2 * dim] * depth, dim, norm=True)

    def forward(self, obs_hist: Tensor, hist_mask: Tensor) -> Tensor:
        """Args: obs_hist (B, H, D_o), hist_mask (B, H) bool. Returns: z_touch (B, dim)."""
        x = torch.cat([obs_hist * hist_mask[..., None], hist_mask[..., None].to(obs_hist.dtype)], -1)
        return self.net(x.flatten(1))


class CNN1dEncoder(nn.Module):
    """Per-step linear embedding -> residual 1-D convolutions over time -> masked mean pooling -> MLP."""

    def __init__(self, obs_dim: int, history: int, dim: int, depth: int, kernel: int = 5):
        """Args: obs_dim: D_o. history: H (unused: any H works). dim: width. depth: conv layers. kernel: odd size."""
        super().__init__()
        self.embed = nn.Linear(obs_dim, dim)
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, kernel, padding=kernel // 2) for _ in range(depth))
        self.head = mlp(dim, dim, dim, norm=True)

    def forward(self, obs_hist: Tensor, hist_mask: Tensor) -> Tensor:
        """Args: obs_hist (B, H, D_o), hist_mask (B, H) bool. Returns: z_touch (B, dim)."""
        m = hist_mask[:, None].to(obs_hist.dtype)  # (B, 1, H)
        x = self.embed(obs_hist).transpose(1, 2) * m  # (B, dim, H); zero padded steps so they add nothing
        for conv in self.convs:
            x = x + nn.functional.gelu(conv(x)) * m
        return self.head(masked_mean(x.transpose(1, 2), hist_mask))


class TransformerEncoder(nn.Module):
    """Per-step embedding + learned positional embedding counted back from now -> Transformer -> masked mean pool."""

    def __init__(self, obs_dim: int, history: int, dim: int, depth: int, heads: int = 4):
        """Args: obs_dim: D_o. history: H. dim: width (divisible by heads). depth: layers. heads: attention heads."""
        super().__init__()
        self.embed = nn.Linear(obs_dim, dim)
        self.pos = nn.Parameter(0.02 * torch.randn(history, dim))  # row 0 = current step, row s = s steps ago
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.net = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, obs_hist: Tensor, hist_mask: Tensor) -> Tensor:
        """Args: obs_hist (B, H, D_o), hist_mask (B, H) bool. Returns: z_touch (B, dim)."""
        H = obs_hist.shape[1]
        x = self.embed(obs_hist) + self.pos[:H].flip(0)  # obs_hist is oldest first
        valid = hist_mask.clone()
        valid[:, -1] = True  # o_t is always real; also guarantees no all-masked row (NaN in attention)
        return self.head(masked_mean(self.net(x, src_key_padding_mask=~valid), valid))

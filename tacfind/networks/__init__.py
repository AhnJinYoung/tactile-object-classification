"""Networks: one file per module, wired together by `model.TacFind`.

Pipeline: touch_encoder (observation history -> z_touch), target_encoder (target cloud -> tokens + z_target),
matcher (touch vs. target -> confidence + z_diff), policy (actor-critic). The two helpers below are shared by them.
"""

from torch import Tensor, nn


def mlp(*dims: int, norm: bool = False) -> nn.Sequential:
    """Linear layers of the given widths with GELU (optionally preceded by LayerNorm) between them.

    Args:
        dims: input width, hidden widths..., output width. The last layer is linear.
        norm: insert a LayerNorm before each hidden activation.
    """
    layers = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(d_out), nn.GELU()] if norm else [nn.GELU()]
    return nn.Sequential(*layers)


def masked_mean(x: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """Mean of `x` over `dim` counting only rows where `mask` is True; zero when none is.

    Args:
        x: (B, T, C) features.
        mask: (B, T) bool.
    Returns:
        (B, C).
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)

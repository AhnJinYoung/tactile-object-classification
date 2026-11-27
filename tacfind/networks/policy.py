"""Actor-critic policy: (o_t, z_touch, z_diff, confidence) -> Gaussian over the 28-D joint-target action, and a value.

The critic may also see privileged simulator state (asymmetric actor-critic).
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from tacfind.networks import mlp


class ActorCritic(nn.Module):
    """Actor MLP -> Gaussian mean with a state-independent learned log-std; critic MLP -> value."""

    def __init__(self, obs_dim: int, act_dim: int, priv_dim: int, dim: int, hidden: Sequence[int] = (256, 256, 128),
                 init_log_std: float = 0.0):
        """Args:
            obs_dim: D_o. act_dim: action size. priv_dim: privileged state size (0 = none).
            dim: width of z_touch and z_diff. hidden: hidden widths of both MLPs. init_log_std: initial log std.
        """
        super().__init__()
        self.priv_dim = priv_dim
        self.norm = nn.LayerNorm(2 * dim)  # latents have no fixed scale, unlike the normalised o_t
        in_dim = obs_dim + 2 * dim + 1
        self.actor, self.critic = mlp(in_dim, *hidden, act_dim), mlp(in_dim + priv_dim, *hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)  # near-zero initial actions

    def forward(self, obs: Tensor, z_touch: Tensor, z_diff: Tensor, conf: Tensor,
                priv: Tensor | None = None) -> tuple[Normal, Tensor]:
        """Args:
            obs: (B, D_o) current observation o_t. z_touch, z_diff: (B, dim). conf: (B,) confidence in [0, 1].
            priv: (B, priv_dim) privileged state, required when priv_dim > 0.
        Returns:
            action distribution with (B, act_dim) mean, value (B,).
        """
        x = torch.cat([obs, self.norm(torch.cat([z_touch, z_diff], -1)), conf[:, None]], -1)
        dist = Normal(self.actor(x), self.log_std.exp().expand(x.shape[0], -1))
        if self.priv_dim:
            x = torch.cat([x, priv], -1)
        return dist, self.critic(x).squeeze(-1)

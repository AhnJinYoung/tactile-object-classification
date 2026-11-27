"""TacFind: the single aggregate that wires the four network modules together.

touch_encoder(obs_hist) -> z_touch; target_encoder(cloud) -> TargetCode; matcher(z_touch, touch_pts, TargetCode)
-> z_diff, confidence, e_touch; policy(o_t, z_touch, z_diff, confidence) -> (action distribution, value).
Each module is built by Hydra from its config group, so swapping an architecture is a config change.
"""

from dataclasses import dataclass

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions import Normal

from tacfind.networks.target_encoder import TargetCode
from tacfind.spec import EnvSpec


@dataclass
class Percept:
    """What the robot currently believes about the touched object.

    Attributes:
        z_touch: (B, dim) spatial-temporal touch embedding.
        z_diff: (B, dim) pooled difference between target and touch evidence.
        conf_logit: (B,) logit of "the touched object is the target".
        e_touch: (B, dim) contrastive embedding of the touched object.
        est_points: (B, K, 3) estimated contact points (m, centred) or None (only `matcher=latent_head`).
    """

    z_touch: Tensor
    z_diff: Tensor
    conf_logit: Tensor
    e_touch: Tensor
    est_points: Tensor | None = None

    @property
    def conf(self) -> Tensor:
        """(B,) confidence in [0, 1]."""
        return torch.sigmoid(self.conf_logit)


class TacFind(nn.Module):
    """Touch encoder + target encoder + cross-modal matcher + actor-critic."""

    def __init__(self, cfg: DictConfig, spec: EnvSpec):
        """Args: cfg: root config (groups touch_encoder, target_encoder, matcher, policy; key history). spec: sizes."""
        super().__init__()
        self.touch_encoder = instantiate(cfg.touch_encoder, obs_dim=spec.obs_dim, history=cfg.history)
        self.target_encoder = instantiate(cfg.target_encoder)
        self.matcher = instantiate(cfg.matcher)
        self.policy = instantiate(cfg.policy, obs_dim=spec.obs_dim, act_dim=spec.act_dim, priv_dim=spec.priv_dim)
        self.trainable: list[str] | None = None  # set by `set_trainable`; None = everything trains

    def encode_target(self, cloud: Tensor) -> TargetCode:
        """Args: cloud (B, N, 6) = [xyz object frame (m), unit normal]. Returns: TargetCode."""
        return self.target_encoder(cloud)

    def perceive(self, obs_hist: Tensor, hist_mask: Tensor, touch_pts: Tensor, touch_mask: Tensor,
                 target: TargetCode) -> Percept:
        """Args:
            obs_hist: (B, H, D_o) oldest first. hist_mask: (B, H) bool, True = real step.
            touch_pts: (B, L, 8) contact points. touch_mask: (B, L) bool. target: TargetCode with B rows.
        """
        z_touch = self.touch_encoder(obs_hist, hist_mask)
        return Percept(z_touch, **self.matcher(z_touch, touch_pts, touch_mask, target))

    def act(self, obs_hist: Tensor, percept: Percept, priv: Tensor | None = None,
            detach_matcher: bool = True) -> tuple[Normal, Tensor]:
        """Policy on o_t = obs_hist[:, -1].

        Args:
            detach_matcher: stop policy-loss gradients into z_diff / confidence (z_touch keeps its gradient path).
        Returns:
            action distribution (mean (B, act_dim)), value (B,).
        """
        z_diff, conf = percept.z_diff, percept.conf
        if detach_matcher:
            z_diff, conf = z_diff.detach(), conf.detach()
        return self.policy(obs_hist[:, -1], percept.z_touch, z_diff, conf, priv)

    def forward(self, obs_hist: Tensor, hist_mask: Tensor, touch_pts: Tensor, touch_mask: Tensor, cloud: Tensor,
                priv: Tensor | None = None, detach_matcher: bool = True) -> tuple[Normal, Tensor, Percept]:
        """encode_target + perceive + act in one call. Returns: (dist, value (B,), Percept)."""
        percept = self.perceive(obs_hist, hist_mask, touch_pts, touch_mask, self.encode_target(cloud))
        return *self.act(obs_hist, percept, priv, detach_matcher), percept

    def set_trainable(self, names: list[str]) -> list[nn.Parameter]:
        """Train only the named sub-modules (dotted paths, e.g. "matcher.embed"); freeze the rest in eval mode.

        The choice persists: a later `model.train()` keeps the frozen parts in eval mode.
        Returns:
            the trainable parameters, for the optimiser.
        """
        self.trainable = list(names)
        self.requires_grad_(False)
        for name in names:
            self.get_submodule(name).requires_grad_(True)
        self.train(self.training)
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> "TacFind":
        """nn.Module.train, except that parts frozen by `set_trainable` stay in eval mode."""
        super().train(mode)
        if mode and self.trainable is not None:
            for module in self.children():
                module.train(False)
            for name in self.trainable:
                self.get_submodule(name).train()
        return self

    def param_counts(self) -> dict[str, int]:
        """Parameters per module (and per matcher stage), plus the total."""
        modules = dict(self.named_children(), **{"matcher.embed": self.matcher.embed,
                                                 "matcher.confidence": self.matcher.confidence})
        counts = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
        return counts | {"total": sum(p.numel() for p in self.parameters())}

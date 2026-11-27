"""Rollout storage: per-step tensors of one iteration on the env device, history windows built by indexing, and GAE.

Sits between `BinEnv` and the update in `trainer.py`. Every observation is stored once, at a "position"; the window
the networks see at step t (o_{t-H+1..t} and the tactile points of those steps) is gathered from the positions by
index, so the window used when acting and the one rebuilt in the update are the same tensors by construction.
Layout per iteration: positions 0..H-2 hold the H-1 observations before step 0 (carried over from the previous
iteration), position t+H-1 holds the observation acted on at step t, and position T+H-1 the observation after the
last step (for the bootstrap value; it becomes step 0 of the next iteration).
"""

import torch
from torch import Tensor


class Rollout:
    """Storage of T steps of N worlds plus the H steps around them.

    Args:
        n_envs: N. steps: T per iteration. history: H. n_lib: object library size (for window labels).
        device: env device.
    """

    def __init__(self, n_envs: int, steps: int, history: int, n_lib: int, device: torch.device):
        self.n, self.T, self.H, self.n_lib, self.device = n_envs, steps, history, n_lib, device
        self.pos: dict[str, Tensor] = {}   # (T + H, N, ...) per observation
        self.step: dict[str, Tensor] = {}  # (T, N, ...) per action
        self.start = torch.zeros(steps + history, n_envs, dtype=torch.long, device=device)  # episode start position
        self.age = (history - 1 - torch.arange(history, device=device)) / max(history - 1, 1)  # (H,) 0 = now

    @staticmethod
    def _put(store: dict, size: int, key: str, idx: int, value: Tensor) -> None:
        if key not in store:
            store[key] = torch.zeros(size, *value.shape, dtype=value.dtype, device=value.device)
        store[key][idx] = value

    def add_obs(self, k: int, **fields: Tensor) -> None:
        """Store the observation at position k (fields: o, pts, valid, first, target, obj, conf_gt, priv)."""
        for key, value in fields.items():
            self._put(self.pos, self.T + self.H, key, k, value)
        self.start[k] = torch.where(fields["first"], k, self.start[k - 1] if k > 0 else 0)

    def add_step(self, t: int, **fields: Tensor) -> None:
        """Store per-action data of step t (action, logp, value, reward, done, trunc, z_touch, z_diff, conf_logit)."""
        for key, value in fields.items():
            self._put(self.step, self.T, key, t, value)

    def advance(self) -> None:
        """Carry the last H positions to the front for the next iteration."""
        T = self.T
        for v in self.pos.values():
            v[:self.H] = v[T:].clone()
        self.start[:self.H] = self.start[T:] - T  # negative = the episode started before the buffer

    def has_contact(self) -> Tensor:
        """(T, N) bool: the window of step t holds at least one valid touch point (cheaper than `window`)."""
        k = torch.arange(self.T + self.H, device=self.device)[:, None]
        last = torch.where(self.pos["valid"].any(-1), k, -1).cummax(0).values  # latest contact position up to k
        end = slice(self.H - 1, self.H - 1 + self.T)
        return last[end] >= torch.maximum(k[:self.T], self.start[end])  # inside the window and the episode

    def window(self, t: Tensor, n: Tensor) -> dict[str, Tensor]:
        """History windows of samples (step t, world n).

        Args:
            t: (B,) long step index in [0, T] (T = the observation after the last step). n: (B,) long world index.
        Returns:
            obs_hist (B, H, D_o) oldest first, hist_mask (B, H) bool, touch_pts (B, H*5, 8) = [pos_base (m),
            normal_base, force (N), age in [0, 1] (0 = now)], touch_mask (B, H*5) bool; zero where masked.
            Labels: touched (B,) library index owning most valid contacts of the window (-1 = none), conf_gt (B,),
            target (B,) library index, priv (B, priv_dim). Masked = before the episode of step t started.
        """
        end = t + self.H - 1
        pos, rows = t[:, None] + torch.arange(self.H, device=self.device), n[:, None]  # (B, H)
        hist_mask = pos >= self.start[end, n][:, None]
        p = self.pos
        touch_mask = p["valid"][pos, rows] & hist_mask[..., None]  # (B, H, 5)
        age = self.age[None, :, None, None].expand(*touch_mask.shape, 1)
        touch_pts = torch.cat([p["pts"][pos, rows], age], -1) * touch_mask[..., None]
        obj = p["obj"][pos, rows].masked_fill(~touch_mask, -1).flatten(1)  # (B, H*5)
        counts = torch.zeros(len(t), self.n_lib, device=self.device).scatter_add_(1, obj.clamp(min=0),
                                                                                 (obj >= 0).float())
        return {
            "obs_hist": p["o"][pos, rows] * hist_mask[..., None], "hist_mask": hist_mask,
            "touch_pts": touch_pts.flatten(1, 2), "touch_mask": touch_mask.flatten(1),
            "touched": torch.where(counts.amax(1) > 0, counts.argmax(1), -1),
            "conf_gt": p["conf_gt"][end, n], "target": p["target"][end, n], "priv": p["priv"][end, n],
        }


def gae(reward: Tensor, value: Tensor, done: Tensor, last_value: Tensor, gamma: float,
        lam: float) -> tuple[Tensor, Tensor]:
    """Generalised advantage estimation.

    Args:
        reward, value: (T, N). done: (T, N) bool, True = the episode ended at this step (no bootstrap across it).
        last_value: (N,) value of the observation after step T-1. gamma: discount. lam: GAE lambda.
    Returns:
        advantages (T, N), returns (T, N) = advantages + value.
    """
    adv, running = torch.zeros_like(reward), torch.zeros_like(last_value)
    for t in reversed(range(reward.shape[0])):  # loop over time only, all worlds at once
        next_value = last_value if t == reward.shape[0] - 1 else value[t + 1]
        live = (~done[t]).to(reward.dtype)
        running = reward[t] + gamma * next_value * live - value[t] + gamma * lam * live * running
        adv[t] = running
    return adv, adv + value

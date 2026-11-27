"""Intermediate visualisation: a rollout video of one world with a confidence overlay, and the metric curves figure.

Rendering runs on the CPU from one world's state (`BinEnv.world_state`); nothing else leaves the env device.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tacfind.envs.render import VideoRecorder, WorldRenderer, overlay  # noqa: E402


class Clip:
    """A video of world `world` over the next `steps` env steps (may span several rollout iterations).

    Args:
        env: the environment. world: index of the rendered world. path: output mp4. steps: frames to record.
    """

    renderer: WorldRenderer | None = None  # one GL context per process, shared by all clips

    def __init__(self, env, world: int, path: Path, steps: int):
        self.env, self.world, self.left = env, world, steps
        self.rec = VideoRecorder(path, fps=1.0 / (env.cfg.timestep * env.cfg.substeps))
        if Clip.renderer is None or Clip.renderer.env is not env:
            if Clip.renderer is not None:
                Clip.renderer.close()
            Clip.renderer = WorldRenderer(env)

    def add(self, conf_gt: float, conf_pred: float, text: str) -> bool:
        """Render the current frame with bars for ground-truth / predicted confidence. Returns True when complete."""
        self.rec.add(overlay(Clip.renderer(self.world), {"conf_gt": conf_gt, "conf_pred": conf_pred}, text))
        self.left -= 1
        return self.left <= 0

    def save(self) -> Path | None:
        """Write the mp4 (None if no frame was recorded)."""
        return self.rec.save() if self.rec.frames else None


def plot_curves(records: list[dict], path: Path) -> None:
    """One panel per `group/name` scalar against environment steps; stage changes as dotted vertical lines.

    Args:
        records: metrics.jsonl rows. path: output png.
    """
    keys = sorted({k for r in records for k, v in r.items() if "/" in k and v is not None})
    if not keys:
        return
    cols = min(4, len(keys))
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    bounds = [r["step"] for a, r in zip(records, records[1:]) if a["stage"] != r["stage"]]
    for ax, key in zip(axes.flat, keys):
        pts = [(r["step"], r[key]) for r in records if r.get(key) is not None]
        ax.plot(*zip(*pts), lw=1.2, marker="." if len(pts) < 20 else None)
        for b in bounds:
            ax.axvline(b, color="0.6", ls=":", lw=0.8)
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.suptitle(" | ".join(dict.fromkeys(r["stage"] for r in records)), fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.04, top=1 - 0.8 / (2.6 * rows), wspace=0.3, hspace=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)

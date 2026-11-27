"""Evaluation: roll out a trained policy (or the scripted explorer) on a library split until enough episodes have
finished, and log success / wrong-pick rates, perception metrics and a video into a new run directory."""

import math
from collections import defaultdict

import torch

from tacfind.train.logger import rates, to_floats
from tacfind.train.stages import Stage
from tacfind.train.trainer import PERCEPTION, Trainer
from tacfind.train.viz import Clip


@torch.no_grad()
def evaluate(tr: Trainer, actor: str, episodes: int, perception: bool, deterministic: bool = True) -> dict:
    """Roll out until `episodes` episodes have finished; log one metrics row (stage "eval") and videos/eval_<actor>.mp4.

    Args:
        tr: trainer holding the env and the (loaded) model. actor: "policy" | "explorer". episodes: finished
            episodes to collect. perception: also score the matcher on the collected windows (needs trained weights).
        deterministic: the policy acts with its mean action.
    Returns:
        the logged record.
    """
    stage = Stage("eval", actor, 0, [], dict.fromkeys(PERCEPTION, 1.0) if perception else {}, 0.0, 1,
                  tr.cfg.perceive_chunk)
    tr.model.set_trainable([])
    tr.deterministic, tr.recompute, tr.augment = deterministic, (False, False), False
    tr.clip = Clip(tr.env, tr.cfg.viz_world, tr.log.dir / "videos" / f"eval_{actor}.mp4", tr.cfg.viz_steps)
    max_iters = math.ceil(tr.env.max_steps / tr.T) * (math.ceil(episodes / tr.N) + 1)  # every world ends by then
    sums, perc = None, defaultdict(list)
    for it in range(max_iters):
        s = tr.collect(stage, it)
        sums = s if sums is None else {k: sums[k] + s[k] for k in s}
        samples = torch.arange(tr.T * tr.N, device=tr.dev)
        for mb in samples.split(tr.cfg.perceive_chunk) if perception else []:
            for k, v in tr.losses(stage, mb // tr.N, mb % tr.N, None, None)[1].items():
                perc[k].append(v)
        tr.env_steps += tr.N * tr.T
        if sums["episodes"] >= episodes:  # one host sync per iteration
            break
    tr._finish_clip()
    scalars = rates(sums) | {k: torch.stack(v).float().nanmean() for k, v in perc.items()}
    return tr.log.log(tr.env_steps, it, "eval", to_floats(scalars, tr.dev))

"""Training stages: parse `train.stages` (configs/train/*.yaml) into `Stage`s and decide what a stage recomputes.

A stage = which actor drives the robot, which sub-modules of `TacFind` train, and the weights of the loss terms.
"""

from dataclasses import dataclass

from omegaconf import DictConfig

from tacfind.networks.matcher import PointTokenizer
from tacfind.networks.model import TacFind

LOSS_TERMS = ("ppo", "contrastive", "uniform", "confidence", "chamfer")


@dataclass
class Stage:
    """One training stage.

    Attributes:
        name: stage name (checkpoint and video names). actor: "explorer" | "policy". iters: iterations.
        train: dotted sub-module names of TacFind that train (aliases expanded). loss: weight per enabled term.
        lr: Adam learning rate. epochs: passes over each rollout. batch_size: samples per minibatch.
        detach_matcher: stop PPO gradients into the matcher outputs.
    """

    name: str
    actor: str
    iters: int
    train: list[str]
    loss: dict[str, float]
    lr: float
    epochs: int
    batch_size: int
    detach_matcher: bool = True

    @property
    def contact_only(self) -> bool:
        """Only losses that need a touch (contrastive, uniform, chamfer): windows without contact add nothing."""
        return set(self.loss) <= {"contrastive", "uniform", "chamfer"}


def matcher_reads_z_touch(model: TacFind) -> bool:
    """True for the `latent` and `latent_head` matchers, whose touch side is z_touch instead of the contact points."""
    return not isinstance(model.matcher.embed.tokenizer, PointTokenizer)


def expand(names: list[str], model: TacFind) -> list[str]:
    """Replace the alias `touch_side` by matcher.embed (+ touch_encoder when the matcher reads z_touch)."""
    out = []
    for name in names:
        out += (["matcher.embed"] + ["touch_encoder"] * matcher_reads_z_touch(model)) if name == "touch_side" else [name]
    return list(dict.fromkeys(out))


def overlaps(trainable: list[str], modules: list[str]) -> bool:
    """True if any trainable sub-module is, contains, or lies inside one of `modules` (dotted names)."""
    return any(a == b or a.startswith(f"{b}.") or b.startswith(f"{a}.") for a in trainable for b in modules)


def matcher_inputs(model: TacFind) -> list[str]:
    """Sub-modules the matcher outputs (z_diff, confidence) depend on."""
    return ["matcher", "target_encoder"] + ["touch_encoder"] * matcher_reads_z_touch(model)


def parse_stages(train_cfg: DictConfig, model: TacFind, optim: DictConfig, only: list[str] | None = None,
                 max_iters: int | None = None) -> list[Stage]:
    """Stages of the training scheme, in config order.

    Args:
        train_cfg: the `train` group (key `stages`: name -> stage). model: for alias expansion.
        optim: defaults of `epochs` and `batch_size` (perception / ppo). only: run only these stage names.
        max_iters: cap on every stage's iters.
    """
    unknown = set(only or []) - set(train_cfg.stages)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; scheme has {list(train_cfg.stages)}")
    stages = []
    for name, s in train_cfg.stages.items():
        loss = {k: float(w) for k, w in s.loss.items() if w}
        if set(loss) - set(LOSS_TERMS) or s.actor not in ("explorer", "policy"):
            raise ValueError(f"stage {name}: bad loss terms {list(loss)} or actor {s.actor}")
        if "ppo" in loss and s.actor != "policy":
            raise ValueError(f"stage {name}: the ppo loss needs actor=policy")
        batch = optim.batch_size.ppo if set(loss) <= {"ppo", "uniform"} else optim.batch_size.perception
        if only is None or name in only:
            stages.append(Stage(name, s.actor, min(s.iters, max_iters or s.iters), expand(list(s.train), model),
                                loss, s.lr, s.get("epochs", optim.epochs), s.get("batch_size", batch),
                                s.get("detach_matcher", True)))
    return stages

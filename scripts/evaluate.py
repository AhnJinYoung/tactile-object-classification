"""Evaluate a trained run (or the scripted explorer) on the train or held-out ("test") objects.

    python scripts/evaluate.py --run runs/20261201-120000_E001_baseline --split test --episodes 200
    python scripts/evaluate.py --actor explorer --split test n_envs=64            # no checkpoint needed

Writes a new run directory runs/<timestamp>_eval_<source>_<split>_<actor>/ (summary.json, metrics.jsonl with one
row of stage "eval", videos/eval_<actor>.mp4). Trailing key=value arguments override the config.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede the mujoco import

from pathlib import Path  # noqa: E402

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from tacfind.train.evaluate import evaluate  # noqa: E402
from tacfind.train.logger import RunLogger  # noqa: E402
from tacfind.train.trainer import Trainer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="trained run directory (required for --actor policy)")
    ap.add_argument("--ckpt", default="last", help="checkpoint name in <run>/ckpt/ (a stage name or last)")
    ap.add_argument("--actor", choices=["policy", "explorer"], default="policy")
    ap.add_argument("--split", choices=["train", "test"], default="test", help="test = held-out library objects")
    ap.add_argument("--episodes", type=int, default=100, help="finished episodes to collect")
    ap.add_argument("--stochastic", action="store_true", help="sample policy actions instead of the mean")
    ap.add_argument("overrides", nargs="*", help="config overrides key=value, e.g. n_envs=256 viz_steps=300")
    args = ap.parse_args()
    if args.run is None and args.actor == "policy":
        ap.error("--actor policy needs --run")
    if args.run:
        cfg = OmegaConf.merge(OmegaConf.load(args.run / "config.yaml"), OmegaConf.from_dotlist(args.overrides))
        cfg.init_from, source = str(args.run / "ckpt" / f"{args.ckpt}.pt"), args.run.name.split("_", 1)[-1]
    else:
        with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
            cfg = compose("config", args.overrides)
        source = "untrained"
    cfg.name = f"eval_{source}_{args.split}_{args.actor}"
    cfg.wandb.enabled = False
    meta = {"kind": "eval", "source_run": args.run.name if args.run else None, "ckpt": args.ckpt if args.run else None,
            "split": args.split, "actor": args.actor, "overrides": args.overrides}
    logger = RunLogger(cfg, meta)
    try:
        tr = Trainer(cfg, logger, split="heldout" if args.split == "test" else "train")
        rec = evaluate(tr, args.actor, args.episodes, perception=args.run is not None,
                       deterministic=not args.stochastic)
        logger.close("done")
    except BaseException:
        logger.close("failed")
        raise
    print({k: v for k, v in rec.items() if v is not None})
    print(f"run directory: {logger.dir}")


if __name__ == "__main__":
    main()

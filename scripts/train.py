"""Training entry point (Hydra): one run = configs/config.yaml + overrides -> runs/<timestamp>_<name>/.

    python scripts/train.py +experiment=E001_baseline
    python scripts/train.py -m history=10,30,60          # sequential sweep in this process (use launch.py on GPUs)
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering of the videos; must precede the mujoco import

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from tacfind.train.logger import RunLogger  # noqa: E402
from tacfind.train.trainer import Trainer  # noqa: E402


def train(cfg: DictConfig, choices: dict | None = None, overrides: list[str] | None = None) -> RunLogger:
    """Run every selected stage of `cfg`; the run directory's summary ends "done" or "failed".

    Args:
        cfg: root config. choices: selected option of every config group. overrides: command-line overrides.
    Returns:
        the run's logger (its `dir` is the run directory).
    """
    logger = RunLogger(cfg, {"kind": "train", "choices": choices or {}, "overrides": overrides or []})
    trainer = None
    try:
        trainer = Trainer(cfg, logger)
        trainer.run()
        logger.close("done")
    except BaseException:
        logger.close("failed")
        raise
    finally:
        if trainer is not None:
            trainer.close()  # live viewer, cpu backend threads (a sweep runs several jobs in one process)
    print(f"run directory: {logger.dir}")
    return logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    hc = HydraConfig.get()
    choices = {k: v for k, v in hc.runtime.choices.items() if not k.startswith("hydra")}
    train(cfg, {"experiment": None} | choices, list(hc.overrides.task))


if __name__ == "__main__":
    main()
